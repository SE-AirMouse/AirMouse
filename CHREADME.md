# 前言
薛尹喆大電神把v0.0改成超讚的v0.1 ouob

# 系統需求
- 已安裝python
- 於系統終端執行 `pip install -r req.txt` 以安裝相關套件


# 使用須知:
    
按下 'esc' 或 'q' 以退出


當 *Searching* 視窗彈出時，將您的手移至相機內

接著，將您想用來控制滑鼠的手的大拇指彎曲，使系統辨識並追蹤您的手。

接著，您會看見*Matching* 視窗，表示系統已辨認出欲控制的手。

接著，保持大拇指彎曲並向四周稍微揮動您的手以利系統獲取您的手的詳細位置資訊。這個步驟將不會持續太久，煩請耐心等候。

當系統成功獲取位置後，*Controlling*視窗將彈出，代表您可以開始使用手來控制鼠標了!


彎曲全部手指會使系統從*Controlling*模式回到 *Searching*模式(此時手會看起來像貓掌一樣)

系統將會自動退出*Controlling*模式如果相機已無法拍攝到擁有控制權的手。

## mouse moving
**thumb:**

Mouse will be dragged by your hand as long as you keep the thumb curved

## mouse left click 
**forefinger:**

Curve it to hold and uncurve it to release

Curve then uncurve it fast to click, just like common mouse
## mouse right click
**middle finger:**

Curve it to hold and uncurve it to release

Curve then uncurve it fast to click, just like common mouse, idk why do you need double right click anyway
## mouse scrolling
**ring finger:**

Curve it then move your hand to scroll.

The mouse can't be dragged while scrolling
## mouse slow moving mode
**little finger:**

Curve it to make dragging/scrolling slow down.

You still have to curve the thumb or ring finger to drag/scroll while in slow moving mode
## mouse left double click 
**forefinger( & middle finger):**

Original method: This can be done by left clicking twice fast.

Another method: Curve forefinger first then curve middle finger to double click.
        
`This additional rule is designed to make you still be able to double click under low fps, which is hard to be done by the original method`
## zooming 
**thumb & ring finger:**

curve both of them then move your hand up/down to zoom in/out the page


# Common Issues:

## camera

`Exception: Unable to access camera, please check README.md for more info`

check if the camera wasnt attached, or the permission got denied 

you can also try changing the `CAM_NUM` inside main.py(line 15) to 1 or 2 (depending on ur camera's id)
