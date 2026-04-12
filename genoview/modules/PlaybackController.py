import cffi
from types import SimpleNamespace

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
        self.selectionStart = 0
        self.selectionEnd = max(0, frameCount - 1)
        self.loopEnabled = False
        self.loopStart = 0
        self.loopEnd = max(0, frameCount - 1)

    @property
    def current_frame(self):
        return int(self.frame) % self.frameCount

    @property
    def selection_range(self):
        start = int(min(self.selectionStart, self.selectionEnd))
        end = int(max(self.selectionStart, self.selectionEnd))
        return start, end

    @property
    def loop_range(self):
        start = int(min(self.loopStart, self.loopEnd))
        end = int(max(self.loopStart, self.loopEnd))
        return start, end

    def _clamp_frame(self, frame):
        return int(min(max(int(frame), 0), self.frameCount - 1))

    def set_current_frame(self, frame):
        self.frame = float(self._clamp_frame(frame))

    def mark_selection_start(self, frame=None):
        self.selectionStart = self._clamp_frame(self.current_frame if frame is None else frame)
        if self.selectionEnd < self.selectionStart:
            self.selectionEnd = self.selectionStart

    def mark_selection_end(self, frame=None):
        self.selectionEnd = self._clamp_frame(self.current_frame if frame is None else frame)
        if self.selectionStart > self.selectionEnd:
            self.selectionStart = self.selectionEnd

    def collapse_selection_to_current_frame(self):
        self.selectionStart = self.current_frame
        self.selectionEnd = self.current_frame

    def set_loop_from_selection(self):
        self.loopStart, self.loopEnd = self.selection_range
        self.loopEnabled = True
        loopStart, loopEnd = self.loop_range
        if self.current_frame < loopStart or self.current_frame > loopEnd:
            self.set_current_frame(loopStart)

    def clear_loop(self):
        self.loopEnabled = False

    def _apply_loop_constraints(self):
        if not self.loopEnabled or self.frameCount <= 0:
            return
        loopStart, loopEnd = self.loop_range
        if loopEnd <= loopStart:
            self.frame = float(loopStart)
            return

        if self.frame < loopStart:
            self.frame = float(loopStart)
            return

        if self.frame > loopEnd + 0.999:
            loopLength = float(loopEnd - loopStart + 1)
            self.frame = float(loopStart) + (self.frame - float(loopStart)) % loopLength

    def update(self, dt):
        if self.playing:
            self.frame = self.frame + self.speeds[self.speedIndex] * dt / self.frameTime
        if not self.loopEnabled:
            self.frame = self.frame % self.frameCount
        self._apply_loop_constraints()
        return self.current_frame

    def get_ui_layout(self, screenWidth, screenHeight, numLabelTracks=1):
        numLabelTracks = max(0, int(numLabelTracks))
        panelHeight = 74 + 20 * numLabelTracks
        panel = Rectangle(20, screenHeight - 20 - panelHeight, screenWidth - 40, panelHeight)
        controlsY = panel.y + 14
        labelX = 40
        labelWidth = 55
        timelineX = 108
        frameReadoutWidth = 92
        rightGap = 16
        readoutGap = 12
        panelRight = panel.x + panel.width
        timelineWidth = max(
            100,
            panelRight - timelineX - frameReadoutWidth - readoutGap - rightGap,
        )
        firstTrackY = controlsY + 36
        trackSpacing = 20
        labelRows = []
        for index in range(numLabelTracks):
            labelRows.append(
                (
                    Rectangle(labelX, firstTrackY + index * trackSpacing - 8, labelWidth, 18),
                    Rectangle(timelineX, firstTrackY + index * trackSpacing, timelineWidth, 10),
                )
            )
        sliderY = firstTrackY + numLabelTracks * trackSpacing + 4
        return SimpleNamespace(
            panel=panel,
            controls_y=controlsY,
            label_rows=labelRows,
            frame_label=Rectangle(labelX, sliderY - 4, labelWidth, 18),
            slider=Rectangle(timelineX, sliderY, timelineWidth, 14),
            frame_readout=Rectangle(panelRight - frameReadoutWidth - rightGap, sliderY - 2, frameReadoutWidth, 18),
        )

    def draw_ui(self, screenWidth, screenHeight, numLabelTracks=1):
        layout = self.get_ui_layout(screenWidth, screenHeight, numLabelTracks=numLabelTracks)
        playbackPanel = layout.panel
        playbackSlider = layout.slider
        controlsY = layout.controls_y

        GuiGroupBox(playbackPanel, b"Playback")

        if GuiButton(Rectangle(30, controlsY, 70, 24), b"Pause" if self.playing else b"Play"):
            self.playing = not self.playing

        if GuiButton(Rectangle(110, controlsY, 55, 24), b"Prev"):
            self.playing = False
            self.set_current_frame((self.current_frame - 1) % self.frameCount)

        if GuiButton(Rectangle(175, controlsY, 55, 24), b"Next"):
            self.playing = False
            self.set_current_frame((self.current_frame + 1) % self.frameCount)

        if GuiButton(Rectangle(240, controlsY, 45, 24), b"Mark A"):
            self.mark_selection_start()

        if GuiButton(Rectangle(295, controlsY, 45, 24), b"Mark B"):
            self.mark_selection_end()

        if GuiButton(Rectangle(350, controlsY, 45, 24), b"Single"):
            self.collapse_selection_to_current_frame()

        if GuiButton(Rectangle(405, controlsY, 55, 24), b"Loop On" if not self.loopEnabled else b"Loop Off"):
            if self.loopEnabled:
                self.clear_loop()
            else:
                self.set_loop_from_selection()

        selectionStart, selectionEnd = self.selection_range
        loopStart, loopEnd = self.loop_range
        GuiLabel(
            Rectangle(470, controlsY + 2, 180, 20),
            b"Sel: %d - %d" % (selectionStart, selectionEnd),
        )
        GuiLabel(
            Rectangle(650, controlsY + 2, 180, 20),
            b"Loop: %d - %d" % (loopStart, loopEnd) if self.loopEnabled else b"Loop: Off",
        )

        for i, speed in enumerate(self.speeds):
            buttonColor = LIGHTGRAY if i == self.speedIndex else RAYWHITE
            DrawRectangleRec(Rectangle(screenWidth - 340 + 60 * i, controlsY, 55, 24), buttonColor)
            if GuiButton(Rectangle(screenWidth - 340 + 60 * i, controlsY, 55, 24), b"%.2fx" % speed):
                self.speedIndex = i

        GuiLabel(
            layout.frame_readout,
            b"%d / %d" % (
                self.current_frame,
                self.frameCount - 1,
            ),
        )

        oldFrame = self.frame
        self.framePtr[0] = self.frame
        GuiSliderBar(
            playbackSlider,
            b"",
            b"",
            self.framePtr,
            0.0,
            float(self.frameCount - 1))

        if abs(self.framePtr[0] - oldFrame) > 1e-4:
            self.frame = float(min(max(self.framePtr[0], 0.0), float(self.frameCount - 1)))
            self._apply_loop_constraints()
