import mouse
import numpy
import pyautogui
import screeninfo


class control:

    def __init__(self):
        self.screenMonitors = screeninfo.get_monitors()
        self.screenSize = numpy.array(
            (self.screenMonitors[0].width, self.screenMonitors[0].height))

    def setPos(self, Pos: numpy.ndarray):
        x, y = Pos
        x = round(x)
        y = round(y)
        x = max(0, min(self.screenSize[0] - 1, x))
        y = max(0, min(self.screenSize[0] - 1, y))
        mouse.move(x, y)

    def addDis(self, Dis):
        newPos = numpy.array(Dis + pyautogui.position())
        self.setPos(newPos)

    def mouseDown(self, button):
        pyautogui.mouseDown(button=button)

    def mouseUp(self, button):
        pyautogui.mouseUp(button=button)

    def mouseDoubleClick(self, button):
        pyautogui.doubleClick(button=button, _pause=False)

    def keyDown(self, button):
        pyautogui.keyDown(button)

    def keyUp(self, button):
        pyautogui.keyUp(button)

    def scroll(self, val):
        # mouse.wheel(int(val / 50))
        pyautogui.scroll(int(val), _pause=False)
        # pyautogui.scroll 記得要 _pause=False 不然會很卡
    
    def hscroll(self, val):
        pyautogui.hscroll(int(val), _pause=False)
        
