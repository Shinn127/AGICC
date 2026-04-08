import cffi

from pyray import Rectangle
from raylib import *


ffi = cffi.FFI()


class PlaybackController:

    def __init__(
        self,
        frameCount,
        frameTime,
        speeds=(0.25, 0.5, 1.0, 1.5, 2.0),
        defaultSpeedIndex=2,
        playing=True):

        self.frameCount = frameCount
        self.frameTime = frameTime
        self.speeds = list(speeds)
        self.speedIndex = defaultSpeedIndex
        self.playing = playing
        self.frame = 0.0
        self.framePtr = ffi.new('float*')

    @property
    def current_frame(self):
        return int(self.frame) % self.frameCount

    def update(self, dt):
        if self.playing:
            self.frame = (self.frame + self.speeds[self.speedIndex] * dt / self.frameTime) % self.frameCount
        return self.current_frame

    def draw_ui(self, screenWidth, screenHeight):
        playbackPanel = Rectangle(20, screenHeight - 80, screenWidth - 40, 60)
        playbackSlider = Rectangle(30, screenHeight - 35, screenWidth - 60, 20)

        GuiGroupBox(playbackPanel, b"Playback")

        if GuiButton(Rectangle(30, screenHeight - 68, 70, 24), b"Pause" if self.playing else b"Play"):
            self.playing = not self.playing

        if GuiButton(Rectangle(110, screenHeight - 68, 55, 24), b"Prev"):
            self.playing = False
            self.frame = float((self.current_frame - 1) % self.frameCount)

        if GuiButton(Rectangle(175, screenHeight - 68, 55, 24), b"Next"):
            self.playing = False
            self.frame = float((self.current_frame + 1) % self.frameCount)

        for i, speed in enumerate(self.speeds):
            buttonColor = LIGHTGRAY if i == self.speedIndex else RAYWHITE
            DrawRectangleRec(Rectangle(250 + 60 * i, screenHeight - 68, 55, 24), buttonColor)
            if GuiButton(Rectangle(250 + 60 * i, screenHeight - 68, 55, 24), b"%.2fx" % speed):
                self.speedIndex = i

        GuiLabel(Rectangle(screenWidth - 170, screenHeight - 68, 140, 20), b"Frame: %d / %d" % (
            self.current_frame,
            self.frameCount - 1))

        oldFrame = self.frame
        self.framePtr[0] = self.frame
        GuiSliderBar(
            playbackSlider,
            b"Frame",
            b"%d" % self.current_frame,
            self.framePtr,
            0.0,
            float(self.frameCount - 1))

        if abs(self.framePtr[0] - oldFrame) > 1e-4:
            self.frame = float(min(max(self.framePtr[0], 0.0), float(self.frameCount - 1)))
