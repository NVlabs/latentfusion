import time
import json
from multiprocessing import Process, Manager
from pathlib import Path
from queue import Full

import cv2
import matplotlib.cm
import pyrealsense2 as rs
import numpy as np
import argparse
from enum import IntEnum

from latentfusion import imutils

FONT = cv2.FONT_HERSHEY_SIMPLEX


class Preset(IntEnum):
    Custom = 0
    Default = 1
    Hand = 2
    HighAccuracy = 3
    HighDensity = 4
    MediumDensity = 5


def get_intrinsics_from_frame(frame):
    intrinsics = frame.profile.as_video_stream_profile().intrinsics
    return {
        'width': intrinsics.width,
        'height': intrinsics.height,
        'intrinsic_matrix':
            [intrinsics.fx, 0, 0,
             0, intrinsics.fy, 0,
             intrinsics.ppx, intrinsics.ppy, 1]
    }


class RealSenseWorker(Process):

    def __init__(self, state, queues, save_dir, use_mask):
        super().__init__()
        self.queues = queues
        self.state = state
        self.use_mask = use_mask

        self.save_dir = save_dir
        if not self.save_dir.exists():
            self.save_dir.mkdir(exist_ok=True, parents=True)

        self.is_saving = False
        self.current_id = 0
        self.current_frame_count = 0

    def _get_next_id(self):
        sub_paths = [x for x in self.save_dir.iterdir() if x.is_dir()]
        if len(sub_paths) > 0:
            next_id = 1 + max([int(s.name) for s in sub_paths])
        else:
            next_id = 0
        return next_id

    def _check_saving(self):
        has_changed = False
        if self.is_saving != self.state['is_saving']:
            has_changed = True
            self.is_saving = self.state['is_saving']
            print(has_changed, self.is_saving, self.state['is_saving'])

        if has_changed:
            if self.is_saving:
                print("STARTING")
                self.is_saving = True
                self.current_id = self._get_next_id()
                self.current_frame_count = 0
            else:
                print("STOPPING")
                self.is_saving = False
                print("Done with sequence {}".format(self.current_id))

        return self.is_saving

    def _init_worker(self):
        # Create a pipeline
        self.pipeline = rs.pipeline()

        # Create a config and configure the pipeline to stream
        #  different resolutions of color and depth streams
        self.config = rs.config()

        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        profile = self.pipeline.start(self.config)
        self.depth_sensor = profile.get_device().first_depth_sensor()

        # Using preset HighAccuracy for recording
        self.depth_sensor.set_option(rs.option.visual_preset, Preset.HighAccuracy)

        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        depth_scale = self.depth_sensor.get_depth_scale()
        self.state['depth_scale'] = depth_scale
        self.depth_scale = depth_scale

        # We will not display the background of objects more than
        #  clipping_distance_in_meters meters away
        clipping_distance_in_meters = 3  # 3 meter
        self.clipping_distance = clipping_distance_in_meters / depth_scale

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        self.align = rs.align(align_to)

    def run(self):
        self._init_worker()

        # decimate_filt = rs.decimation_filter()
        # decimate_filt.set_option(rs.option.filter_magnitude, 2 ** 1)
        temporal_filt = rs.temporal_filter()
        spatial_filt = rs.spatial_filter()
        # threshold_filt = rs.threshold_filter()
        filters = [
            # threshold_filt,
            spatial_filt,
            temporal_filt,
        ]

        last_frame = time.time()
        fps = 5

        # Streaming loop
        try:
            while self.state['is_running']:
                self._check_saving()
                # Get frameset of color and depth
                frames = self.pipeline.wait_for_frames()

                if time.time() - last_frame < 1.0 / fps:
                    continue

                # Align the depth frame to color frame
                aligned_frames = self.align.process(frames)

                # Get aligned frames
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                # Validate that both frames are valid
                if not depth_frame or not color_frame:
                    continue

                for filter in filters:
                    depth_frame = filter.process(depth_frame)

                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                if self.use_mask:
                    mask_image = imutils.mask_chroma(color_image, (30, 100, 80), (70, 255, 255),
                                                     use_bgr=True)
                    mask_image = mask_image & ((depth_image * self.depth_scale) < 1.0)
                    mask_image = imutils.keep_largest_object(mask_image)
                else:
                    mask_image = None

                if self.is_saving:
                    save_dir = self.save_dir / f'{self.current_id:02d}'
                else:
                    save_dir = None

                if self.current_frame_count == 0:
                    intrinsics = get_intrinsics_from_frame(color_frame)
                else:
                    intrinsics = None

                # depth_image[depth_image * self.state['depth_scale'] > 1.5] = 0

                payload = (self.current_id, self.current_frame_count,
                           depth_image, color_image, mask_image, intrinsics, save_dir)
                for queue in self.queues:
                    try:
                        queue.put_nowait(payload)
                    except Full:
                        print("warning: Queue is full")
                        continue

                if self.is_saving:
                    self.current_frame_count += 1
        finally:
            self.pipeline.stop()


class SaverWorker(Process):

    def __init__(self, state, queue):
        super().__init__()
        self.queue = queue
        self.state = state

    def print(self, s):
        print(f"[{self.queue.qsize()}] {s}")

    def run(self):
        while self.state['is_running']:
            _, frame_count, depth_image, color_image, mask_image, intrinsics, save_dir = self.queue.get()
            if save_dir is None:
                continue

            depth_dir = save_dir / 'depth'
            color_dir = save_dir / 'color'
            if not depth_dir.exists():
                depth_dir.mkdir(parents=True)
            if not color_dir.exists():
                color_dir.mkdir(parents=True)

            depth_path = depth_dir / f'{frame_count:06d}.png'
            self.print(f"Saving {depth_path!s}")
            cv2.imwrite(str(depth_path), depth_image)

            color_path = color_dir / f'{frame_count:06d}.jpg'
            self.print(f"Saving {color_path!s}")
            cv2.imwrite(str(color_path), color_image)

            if mask_image is not None:
                mask_dir = save_dir / 'mask'
                if not mask_dir.exists():
                    mask_dir.mkdir(parents=True)
                mask_path = mask_dir / f'{frame_count:06d}.png'
                self.print(f"Saving {mask_path!s}")
                cv2.imwrite(str(mask_path), mask_image.astype(np.uint8) * 255)

            if intrinsics is not None:
                intrinsics_path = save_dir / 'intrinsics.json'
                self.print(f"Saving {intrinsics_path!s}")
                with open(intrinsics_path, 'w') as f:
                    json.dump(intrinsics, f, indent=2)


cmap = matplotlib.cm.get_cmap('magma')


def colorize(array):
    return (cmap(array)[:, :, [2, 1, 0]] * 255.0).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(
        description="Realsense Recorder. Please select one of the optional arguments")
    parser.add_argument("--save-dir", required=True, type=Path,
                        help="set output folder")
    parser.add_argument("--no-mask", action='store_true')
    args = parser.parse_args()

    with Manager() as manager:
        state = manager.dict({
            'is_running': True,
            'is_saving': False,
        })
        save_queue = manager.Queue(maxsize=4)
        show_queue = manager.Queue(maxsize=4)
        rs_worker = RealSenseWorker(state, [save_queue, show_queue], args.save_dir, use_mask=not args.no_mask)
        saver_worker = SaverWorker(state, save_queue)
        rs_worker.start()
        saver_worker.start()

        try:
            while state['is_running']:
                seq_id, frame_count, depth_image, color_image, mask_image, intrinsics, _ = show_queue.get()
                # color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
                depth_scale = state['depth_scale']
                depth_viz = colorize(depth_image.astype(float) * depth_scale)
                frame_images = [color_image, depth_viz]

                if not args.no_mask:
                    mask_viz = colorize(mask_image.astype(float) * 255.0)
                    frame_images.append(mask_viz)
                viz = np.hstack(frame_images)
                cv2.putText(viz, f'{seq_id} {frame_count}', (10, 450), FONT, 2, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow('RealSense', viz)
                q = cv2.waitKey(1)

                if q == ord('q'):
                    print('Shutting down!')
                    cv2.destroyAllWindows()
                    state['is_running'] = False
                if q == ord(' '):
                    state['is_saving'] = not state['is_saving']
        except KeyboardInterrupt:
            print("Shutting down")

        rs_worker.join()


if __name__ == "__main__":
    main()
