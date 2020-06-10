from pathlib import Path

import av
import imageio
import numpy as np
import structlog
import torch
from tqdm.notebook import tqdm


logger = structlog.get_logger(__name__)


def read_metadata(path: Path):
    with av.open(str(path)) as container:
        video_stream = [s for s in container.streams if s.type == 'video'][0]
        metadata = {
            'frame_rate': float(video_stream.average_rate),
            'height': video_stream.height,
            'width': video_stream.width,
            'duration': float(video_stream.time_base * video_stream.duration),
            'num_frames': video_stream.frames,
        }

    return metadata


def save_video(path, frames, video_rate, waveform=None, audio_rate=48000):
    if torch.is_tensor(frames):
        frames = frames.permute(0, 2, 3, 1).numpy()
    if np.issubdtype(frames.dtype, np.floating):
        frames = (frames * 255.0).astype(np.uint8)

    if waveform is not None:
        if torch.is_tensor(waveform):
            waveform = waveform.numpy()
        if waveform.ndim == 1:
            waveform = waveform[None, :]
        waveform = waveform.astype('<f4')

    container = av.open(path, 'w')
    video_stream = container.add_stream("h264", video_rate)
    video_stream.height = frames.shape[1]
    video_stream.width = frames.shape[2]
    audio_stream = None
    if waveform is not None:
        audio_stream = container.add_stream("aac", audio_rate)

    try:
        for i, frame in enumerate(tqdm(frames, desc='saving video')):
            video_frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
            for packet in video_stream.encode(video_frame):
                container.mux(packet)

            if audio_stream:
                num_samples = audio_rate // video_rate
                audio_start = i * num_samples
                audio_stop = audio_start + num_samples
                audio_frame = av.AudioFrame.from_ndarray(waveform[:, audio_start:audio_stop],
                                                         format='fltp', layout='mono')
                audio_frame.rate = audio_rate
                for packet in audio_stream.encode(audio_frame):
                    container.mux(packet)
        # Flush packets that might be left in encoders.
        while True:
            for packet in video_stream.encode(None):
                container.mux(packet)
            else:
                break
        if audio_stream:
            while True:
                for packet in audio_stream.encode(None):
                    container.mux(packet)
                else:
                    break
    finally:
        container.close()


def save_frames(frames, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    for i, frame in enumerate(tqdm(frames)):
        imageio.imsave(save_dir / f'{i:04d}.jpg', (frame * 255).astype(np.uint8))


class PyAVWriter(object):

    def __init__(self, path, video_rate, audio_rate=16000, write_audio=False, crf=0):
        self.path = path
        self.write_audio = write_audio
        self.video_rate = video_rate
        self.audio_rate = audio_rate
        self.crf = crf

        self.container = None
        self.video_stream = None
        self.audio_stream = None

    def __enter__(self):
        self.container = av.open(str(self.path), 'w')
        self.video_stream = self.container.add_stream("h264", self.video_rate)
        self.video_stream.options = {'crf': str(self.crf)}
        self.audio_stream = self.container.add_stream("aac", self.audio_rate)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        while True:
            for packet in self.video_stream.encode(None):
                self.container.mux(packet)
            else:
                break
        if self.audio_stream:
            while True:
                for packet in self.audio_stream.encode(None):
                    self.container.mux(packet)
                else:
                    break

        self.container.close()

    def put_frame(self, image, waveform=None):
        if self.container is None:
            raise RuntimeError("Writer is not active")

        if torch.is_tensor(image):
            image = image.permute(1, 2, 0).numpy()
        if np.issubdtype(image.dtype, np.floating):
            image = (image * 255.0).astype(np.uint8)

        if waveform is not None:
            if torch.is_tensor(waveform):
                waveform = waveform.numpy()
            if waveform.ndim == 1:
                waveform = waveform[None, :]
            waveform = waveform.astype('<f4')

        self.video_stream.height = image.shape[0]
        self.video_stream.width = image.shape[1]

        video_frame = av.VideoFrame.from_ndarray(image, format='rgb24')
        for packet in self.video_stream.encode(video_frame):
            self.container.mux(packet)

        if self.audio_stream and waveform is not None:
            audio_frame = av.AudioFrame.from_ndarray(waveform, format='fltp', layout='mono')
            audio_frame.rate = self.audio_rate
            for packet in self.audio_stream.encode(audio_frame):
                self.container.mux(packet)


class PyAVReader(object):
    """
    Read images from the frames of a standard video file into an
    iterable object that returns images as numpy arrays.
    Parameters
    """

    def __init__(self, file, toc=None, format=None):
        if not hasattr(file, 'read'):
            file = str(file)
        self.file = file
        self.format = format
        self._container = None

        with av.open(self.file, format=self.format) as container:
            stream = [s for s in container.streams if s.type == 'video'][0]

            # Build a toc
            if toc is None:
                packet_lengths = []
                packet_ts = []
                for packet in container.demux(stream):
                    if packet.stream.type == 'video':
                        decoded = packet.decode()
                        if len(decoded) > 0:
                            packet_lengths.append(len(decoded))
                            packet_ts.append(decoded[0].pts)
                self._toc = {
                    'lengths': packet_lengths,
                    'ts': packet_ts,
                }
            else:
                self._toc = toc

            self._toc_cumsum = np.cumsum(self.toc['lengths'])
            self._len = self._toc_cumsum[-1]

            # PyAV always returns frames in color, and we make that
            # assumption in get_frame() later below, so 3 is hardcoded here:
            self._im_sz = stream.height, stream.width, 3
            self._time_base = stream.time_base
            self.rate = stream.average_rate

        self._load_fresh_file()

    @staticmethod
    def _next_video_packet(container_iter):
        for packet in container_iter:
            if packet.stream.type == 'video':
                decoded = packet.decode()
                if len(decoded) > 0:
                    return decoded

        raise ValueError("Could not find any video packets.")

    def _load_fresh_file(self):
        if self._container is not None:
            self._container.close()

        if hasattr(self.file, 'seek'):
            self.file.seek(0)

        self._container = av.open(self.file, format=self.format)
        demux = self._container.demux(self._video_stream)
        self._current_packet = self._next_video_packet(demux)
        self._current_packet_no = 0

    @property
    def _video_stream(self):
        return [s for s in self._container.streams if s.type == 'video'][0]

    def __len__(self):
        return self._len

    def __del__(self):
        if self._container is not None:
            self._container.close()

    def __getitem__(self, item):
        if isinstance(item, int):
            item = slice(item, item + 1)

        if item.start < 0 or item.start >= len(self):
            raise IndexError(f"start index ({item.start}) out of range")

        if item.stop < 0 or item.stop > len(self):
            raise IndexError(f"stop index ({item.stop}) out of range")

        return np.stack([self.get_frame(i) for i in range(item.start, item.stop)])

    @property
    def frame_shape(self):
        return self._im_sz

    @property
    def toc(self):
        return self._toc

    def get_frame(self, j):
        # Find the packet this frame is in.
        packet_no = self._toc_cumsum.searchsorted(j, side='right')
        self._seek_packet(packet_no)
        # Find the location of the frame within the packet.
        if packet_no == 0:
            loc = j
        else:
            loc = j - self._toc_cumsum[packet_no - 1]
        frame = self._current_packet[loc]  # av.VideoFrame

        return frame.to_ndarray(format='rgb24')

    def _seek_packet(self, packet_no):
        """Advance through the container generator until we get the packet
        we want. Store that packet in selfpp._current_packet."""
        packet_ts = self.toc['ts'][packet_no]
        # Only seek when needed.
        if packet_no == self._current_packet_no:
            return
        elif (packet_no < self._current_packet_no
                or packet_no > self._current_packet_no + 1):
            self._container.seek(packet_ts, stream=self._video_stream)

        demux = self._container.demux(self._video_stream)
        self._current_packet = self._next_video_packet(demux)
        while self._current_packet[0].pts < packet_ts:
            self._current_packet = self._next_video_packet(demux)

        self._current_packet_no = packet_no
