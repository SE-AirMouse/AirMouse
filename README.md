# Introduction
this is the source code of airmouse, we will keep updating this till the science exhibition ends.

air mouse v0.1 is the next version of the airmouse v0.0
"supposed" to be more accurate and handy

...

don't blame me if I mess it up instead...

click to jump to [Chinese(Trad.) readme.md](CHREADME.md)

# before use
- python required
- install requirements first by runing `pip install -r req.txt` in command line


# how to use:
    
press 'esc' or 'q' to quit


When *Searching* window pops, move your hand into the area that camera can capture.

Then, curve the thumb of the hand you want to use to make system recognize.

After that, *Matching* window should pop up, meaning that it has found your curving hand.

While Matching, hold your thumb still and move your hand a bit to make system get your hand's position more precisely. This step won't take too long.

After the system get your hand position, *Controlling* window will pop up, meaning that you can start controlling the cursor by your hand!


curve all fingers to quit *Controlling* mode and back to *Searching*(looks like a cat's paw)

the system will also quit *Controlling* mode automatically when the camera can't detect any hand within it's range.

## mouse moving
**thumb:**
![空中滑鼠](https://github.com/Kevin110026/airmouse-v0.1/assets/131368612/07b40efc-ab69-474f-86ae-c3914ceeff9c)

Mouse will be dragged by your hand as long as you keep the thumb curved

## mouse left click 
**forefinger:**
![空中滑鼠 (1)](https://github.com/Kevin110026/airmouse-v0.1/assets/131368612/bca1da17-3192-429d-9e1d-a1b71815f14e)

Curve it to hold and uncurve it to release

Curve then uncurve it fast to click, just like common mouse
## mouse left double click 
**forefinger( & middle finger):**
![空中滑鼠 (2)](https://github.com/Kevin110026/airmouse-v0.1/assets/131368612/f96d36cb-52e6-4c5d-96ba-c26d7464380e)

Original method: This can be done by left clicking twice fast.

Another method: Curve forefinger first then curve middle finger to double click.
        
`This additional rule is designed to make you still be able to double click under low fps, which is hard to be done by the original method`
## mouse right click
**middle finger:**
![空中滑鼠 (3)](https://github.com/Kevin110026/airmouse-v0.1/assets/131368612/c285683e-48cd-4792-a5b3-0db260c75149)

Curve it to hold and uncurve it to release

Curve then uncurve it fast to click, just like common mouse, idk why do you need double right click anyway
## mouse scrolling
**ring finger:**
![空中滑鼠 (4)](https://github.com/Kevin110026/airmouse-v0.1/assets/131368612/8e66b27f-0ecf-4ff5-bd3c-6037c0e5423f)

Curve it then move your hand to scroll.

The mouse can't be dragged while scrolling
## zooming 
**thumb & ring finger:**
![空中滑鼠 (6)](https://github.com/Kevin110026/airmouse-v0.1/assets/131368612/ed3a037c-36f7-4045-b720-33fad647965d)

curve both of them then move your hand up/down to zoom in/out the page
## sensitive adjusting 
**middle & ring finger:**
![空中滑鼠 (5)](https://github.com/Kevin110026/airmouse-v0.1/assets/131368612/a2851667-8b74-46e7-8575-cad973fd9fa0)

curve both of them then move your hand up/down to adjust mouse sensitive
## mouse slow mode
**little finger:**
![空中滑鼠 (7)](https://github.com/Kevin110026/airmouse-v0.1/assets/131368612/a72c7181-3fe1-4471-8a16-da0a778c84fd)

Curve little finger while doing other actions to make them slower.

Actions include moving, scrolling, zooming, sensitive adjusting.



# Common Issues:

## camera

`Exception: Unable to access camera, please check README.md for more info`

check if the camera wasnt attached, or the permission got denied 

you can also try changing the `CAM_NUM` inside main.py(line 15) to 1 or 2 (depending on ur camera's id)
