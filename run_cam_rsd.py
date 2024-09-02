import argparse
import cv2
import matplotlib
import numpy as np
import os
import torch
import pyrealsense2 as rs

from depth_anything_v2.dpt import DepthAnythingV2

def add_text_to_image(image, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(255, 255, 255), thickness=2):
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = position[0] - text_size[0]
    text_y = position[1] + text_size[1]
    cv2.putText(image, text, (text_x, text_y), font, font_scale, color, thickness)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./vis_video_depth')
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    margin_width = 10
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    pipe = rs.pipeline()
    cfg  = rs.config()
    cfg.enable_stream(rs.stream.depth, 640,480, rs.format.z16, 30)
    pipe.start(cfg)
    
    webcam = cv2.VideoCapture(6)
    frame_width, frame_height = int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH)), int(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(webcam.get(cv2.CAP_PROP_FPS))

    # Create empty margins
    vertical_empty_margin = np.ones((frame_height, frame_width//2, 3), dtype=np.uint8) * 255
    split_vertical = np.ones((frame_height, margin_width, 3), dtype=np.uint8) * 255
    split_horizontal = np.ones((margin_width, (frame_width*2)+margin_width, 3), dtype=np.uint8) * 255
    
    while webcam.isOpened():
        frame = pipe.wait_for_frames()
        depth_frame = frame.get_depth_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_cm = cv2.applyColorMap(cv2.convertScaleAbs(depth_image,
                                     alpha = 0.5), cv2.COLORMAP_JET)
        depth_cm = cv2.resize(depth_cm, (frame_width, frame_height))
        ret, raw_frame = webcam.read()
        if not ret:
            break
        
        depth = depth_anything.infer_image(raw_frame, args.input_size)
        
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        
        if args.grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

        add_text_to_image(raw_frame, "Raw Image", (frame_width - 10, 30))
        add_text_to_image(depth, "Depth Anything V2", (frame_width - 10, 30))
        add_text_to_image(depth_cm, "Realsense D455", (frame_width - 10, 30))

        if args.pred_only:
            cv2.imshow('Depth Prediction', depth)
        else:
            combined_depth = cv2.hconcat([depth, split_vertical, depth_cm])
            combined_raw = cv2.hconcat([vertical_empty_margin, raw_frame, split_vertical, vertical_empty_margin])
            combined_frame= cv2.vconcat([combined_depth,split_horizontal, combined_raw])
            cv2.imshow('Predictions', combined_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    webcam.release()
    cv2.destroyAllWindows()
