import math

import drawsvg as draw
import os
import numpy as np

def direcs_in_circle(lumps_per_quad):
    fac = math.pi/(lumps_per_quad*2)
    rr = 10
    return [ (int(rr*math.cos(n*fac)), int(rr*math.sin(n*fac))) for n in range(lumps_per_quad*2) ]



zero_vec =(0,0)
direcs =  [(1,0), (2.1,1), (0.89, 1.0), (1, 2.04), (0,1), (-1.022, 1.997), (-1.07, 1.05), (-2.002, 1.08)]

direcs =  [(1,0), (2,1), (1,1), (0,1)]
direcs =  [(1,0), (2,1), (1,1), (0,1), (1,-1)]
direcs =  [(1,0), (2,1), (1,1), (0,1), (1,-1), (1,2)]
direcs = [(1,0),(0,1)] + [(np.random.normal(), np.random.normal()) for i in range(4)]
direcs =  [(1,0), (2,1), (1,1), (1,2), (0,1), (-1,2),(-1,1), (-2,1)]
direcs =  [(2,0), (2, 1), (1.5,1.5), (1,2), (0,2), (-1,2),(-1.5,1.5), (-2,1)]
direcs =  [(4,0), (4, 2), (3,3), (2,4), (0,4), (-2,4), (-3,3), (-4,2)]
direcs =  [(4,0), (4, 2), (3,3), (2,4), (0,4), (-2,4), (-3,3)] # Has double blobs
direcs =  [(1,0), (2/3, 1/3), (1,1), (1/3.101,2/3.1), (0,1), (-1/3.2,2/3.2),(-1,1), (-2/3.3,1/3.3)] # Broken??
direcs =  [(4,0), (0,4),
           (4, 2), (2,4), (-2,4), (4,-2),
           (4,1), (4,-1), (1,4), (1,-4),
           (3,3), (-3,3),
           ] # Has many double blobs and a nice striping
direcs = [(4,0),(4,1),(4,3),(3,3),(3,4),(1,4),
          (0,4),(-1,4),(-3,4),(-3,3),(-4,3),(-4,1),
          ]
direcs = [(4,0),(4,2),(4,5),(3,6),(3,8),(1,9),
          (0,4),(-1,4),(-3,4),(-3,3),(-4,3),(-4,1),
          ]
direcs = [(4,0),(4,2),(4,5),(3,6),(3,8),(1,9),
          (0,4),(-1,4),(-3,4),(-3,3),(-4,3),(-4,1),
          (1,2),(1,3),(1,4),(1,5),(1,6),
          ] # Was slow to plot with OLD duplicate removal alg, but fast with the new one.
direcs = direcs_in_circle(5)
direcs = direcs_in_circle(8)

width = 500
height = 500
d = draw.Drawing(width, height, origin='top-left')

d.append(draw.Lines(width*0.01, height*0.01,
                     width*0.01, height*0.99,
                     width*0.99, height*0.99,
                    width*0.99, height*0.01,
                    close=True,
            fill="#ffffff",#fill='#eeee00',
            stroke='black'))


def sum_vecs(v1,v2):
    return tuple( x1+x2 for x1,x2 in zip(v1,v2) )

test_sum = sum_vecs((3,4),(5,6))
print(f"Sum vecs (3,4)+(5,6) = {test_sum}")
red = [ zero_vec ]
blue =  [ direcs[0] ]



print(f"red is {red}")
print(f"blue is {blue}")


def NEW_remove_duplicates_from(red, blue):
    print("Removing duplucates (NEW)")
    from collections import Counter
    red_counts_dict = Counter(red)
    blue_counts_dict = Counter(blue)
    all_count_dict = Counter(red+blue)
    red.clear()
    blue.clear()
    for val, counts in all_count_dict.items():
        if val in red_counts_dict and val in blue_counts_dict:
            # it is in both so some consolidation is required
            nr = red_counts_dict[val]
            nb = blue_counts_dict[val]
            count_red = max(nr-nb, 0)
            count_blue = max(nb-nr, 0)
        elif val in red_counts_dict:
            # it is only in red so preserve red
            count_red = counts
            count_blue = 0
        elif val in blue_counts_dict:
            # it is only in blue so preserve blue
            count_red = 0
            count_blue = counts
        else:
            # should not get here
            raise Exception("Something went wrong!")
        assert count_red == 0 or count_blue == 0
        for i in range(count_red):
            red.append(val)
        for i in range(count_blue):
            blue.append(val)

def OLD_remove_duplicates_from(red, blue):
    print("Removing duplucates (OLD)")
    try_again = True
    while try_again:
        try_again = False
        for i in red:
            if i in blue:
                red.remove(i)
                blue.remove(i)
                try_again = True

            if try_again:
                break

        for i in blue:
            if i in red:
                red.remove(i)
                blue.remove(i)
                try_again = True

            if try_again:
                break


for nd,direc in enumerate(direcs[1:]): # Must begin [1: as we already have the first element in red and blue
    print(f"Adding direc is {direc}")
    new_red = [ sum_vecs(v,direc) for v in blue ]
    new_blue = [ sum_vecs(v,direc) for v in red ]
    red = red + new_red
    blue = blue + new_blue
    NEW_remove_duplicates_from(red, blue)

    print(f" ... added direc {direc} which is {nd+1} of {len(direcs[1:])}")
    #print(f"red ={red}")
    #print(f"blue={blue}")





all_x = [v[0] for v in red + blue]
all_y = [v[1] for v in red + blue]
x_min = min( all_x )
x_max = max( all_x )
y_min = min( all_y )
y_max = max( all_y )
x_range = max(x_max-x_min, 1)
y_range = max(y_max-y_min, 1)
print(f"x range {x_min} to {x_max} and y range {y_min} to {y_max}")

def my_colour(n):
    cols = ["black", "red", "blue","green",
            "magenta","pink","orange","gray",
            "pink", "light_green", "purple"]
    if n<0 or n>=len(cols):
        n=0
    return cols[n]
for nv, v in enumerate(red):
    for nd,direc in enumerate(direcs):
        sf = len(direcs)
        x1 = (v[0]-sf*direc[0]-x_min)/x_range # in [0,1]
        y1 = (v[1]-sf*direc[1]-y_min)/y_range # in [0,1]
        x2 = (v[0]+sf*direc[0]-x_min)/x_range # in [0,1]
        y2 = (v[1]+sf*direc[1]-y_min)/y_range # in [0,1]

        x1 = width * (0.05 * (1 - x1) + 0.95 * x1)  # in [0, width]
        y1 = height * (0.05 * (1 - y1) + 0.95 * y1)  # in [0, width]
        x2 = width * (0.05 * (1 - x2) + 0.95 * x2)  # in [0, width]
        y2 = height * (0.05 * (1 - y2) + 0.95 * y2)  # in [0, width]

        if nv in [4, 19]:
             d.append(draw.Lines(x1,y1, x2,y2,
                            stroke=my_colour(nd),
                            #stroke='black',
                            ))


for colour, vertices in ( ('red',red), ('blue',blue)):
    for v in set(vertices):
        x = (v[0]-x_min)/x_range # in [0,1]
        y = (v[1]-y_min)/y_range # in [0,1]

        x = width * (0.05 * (1 - x) + 0.95 * x)  # in [0, width]
        y = height * (0.05 * (1 - y) + 0.95 * y)  # in [0, width]

        r = min(width, height)/2/50/2
        count = vertices.count(v)
        r=r*math.sqrt(count)

        #print(f"Drawing a radius {r} {colour} circle at ({x}, {y}) ")
        d.append(draw.Circle(x, y, r, fill=colour,
                             #stroke_width=1,
                             #stroke='black'
                             ))

    """
    d.append(draw.Lines(0, 45,
                     70, 49,
                     95, -49,
                    -90, -40,
                    close=True,
            fill='#eeee00',
            stroke='black'))
    """



d.save_svg('example.svg')
print("In ")
print(os.getcwd())
print("created example.svg")