from __future__ import with_statement

import numpy as N
import os
import optparse
import gtk, gobject
# import _C_me_gold
import _me_gold

# TODO: change this import * statement
from pylab import *

class YCbCrIter:
    def __init__(self, file, width=352, height=288):
        self.file = file
        self.width = width
        self.height = height
        self.Y_NumElements = width * height
        self.C_NumElements = (width/2) * (height/2) # Assuming 4:2:0
        self.frame_size = self.Y_NumElements + self.C_NumElements * 2
        self.position = 0

    # Returns the next frame of the movie in a tuple (Y, Cb, Cr)
    def next(self):
        if self.file.tell() + self.frame_size > os.path.getsize(self.file.name):
            raise StopIteration
        try:
            data = N.fromfile(file=self.file, dtype=N.uint8, count=self.frame_size)
        except MemoryError: # Reached EOF or didn't read as many bytes as we thought we would
            raise StopIteration 

        self.position += 1

        # TODO: The following way of splitting the array Seems rather ghetto, I think I'm forgetting some python tricks to do this better
        # Could return a 3 dimensional numpy array but thats no good b/c of chroma downsampling
        Y = data[0 : self.Y_NumElements]
        Y.shape = (self.height, self.width)

        Cb = data[self.Y_NumElements : self.Y_NumElements + self.C_NumElements]
        Cb.shape = (self.height/2, self.width/2)

        Cr = data[self.Y_NumElements + self.C_NumElements : ]
        Cr.shape = (self.height/2, self.width/2)

        return (Y, Cb, Cr)

    def seek_frame(self, frame_number):
        self.file.seek(frame_number * self.frame_size, 0)
        self.position = frame_number
        return

    def tell_frame(self):
        return self.frame_number

    def __iter__(self):
        return self


def main():
    op = optparse.OptionParser()
    op.add_option('--verbose', '-v', action='store_true')
    op.add_option('--output-dir', '-o')
    op.add_option('--frame', '-f', action='store', type='int', dest='frame_number', default=0)
    op.add_option('--rawvideo', action='store_true', default=True) 
    op.add_option('--visualize', action='store_true', default=True)
    options, arguments = op.parse_args()
    
    if (len(arguments) == 0):
        print 'No file specified, exiting.'
        exit()
    
    for fp in arguments:
        with open(fp, 'rb') as file:
            me_video(file, options)


def me_video(file, options): #TODO: Again, rawvideo is default to true which is not right
    # TODO: This is definitely not the right way to do this, fix it.
    global previous_frame, frameNumber
    # Set up the plot environment
    if options.visualize:
        fig = figure()
        axcf = subplot(2, 2, 1)
        axis('off')
        axrf = subplot(2, 2, 2, sharex=axcf, sharey = axcf)
        # axrf = subplot(2, 2, 2)
        axis('off')
        axres = subplot(2, 2, 3, sharex=axcf, sharey = axcf)
        # axres = subplot(2, 2, 3)
        axis('off')
        axnewres = subplot(2, 2, 4, sharex=axcf, sharey = axcf)
        # axnewres = subplot(2, 2, 4)
        axis('off')

        canvas = fig.canvas
        fig.subplots_adjust(left=0.1, bottom=0.1, wspace=0.1, hspace=0.1)
   
    # Raw data iterator
    if options.rawvideo:
        video = YCbCrIter(file)
    else:
        return # TODO: add support for non-ycbcr files


    video.seek_frame(options.frame_number)


    previous_frame = video.next() # Load the first frame

    fwidth = 352
    fheight = 288
    if options.visualize:
        # ion()
        axes(axcf)
        imcf = axcf.imshow(previous_frame[0], cmap=cm.gray, vmin=0, vmax=255, interpolation='nearest') # show the Y component
        qartist = axcf.quiver(N.arange(fwidth/16) * 16 + 8, N.arange(fheight/16) * 16 + 8, N.ones((fheight/16, fwidth/16)), N.ones((fheight/16, fwidth/16)), color='r', scale=1, units='x', angles='xy', width=fwidth*0.003)
        axes(axrf)
        imrf = axrf.imshow(previous_frame[0], cmap=cm.gray, vmin=0, vmax=255, interpolation='nearest') # show the Y component
        axes(axres)
        # imres = axres.imshow(previous_frame[0], cmap=cm.gray, vmin=-255, vmax=255, animated=True) # show the Y component
        imres = axres.imshow(previous_frame[0], cmap=cm.gray, vmin=-64, vmax=64, interpolation='nearest') # show the Y component
        axes(axnewres)
        # imnewres = axnewres.imshow(previous_frame[0], cmap=cm.gray, vmin=-255, vmax=255, animated=True) # show the Y component
        imnewres = axnewres.imshow(previous_frame[0], cmap=cm.gray, vmin=-64, vmax=64, interpolation='nearest') # show the Y component
        canvas.draw()

    frameNumber = options.frame_number+1
    # for frame in video:

    def update(video, options):
        global previous_frame, frameNumber
        print 'updating'
        frame = video.next()
        frameNumber = frameNumber + 1
        axes(axcf)
        # cla()
        imcf.set_data(frame[0])
        # (mvx_gold, mvy_gold, sad_gold, res_gold) = _C_me_gold.me_gold(previous_frame[0], frame[0])
        (mvx_gold, mvy_gold, res_gold) = _me_gold.me_gold(previous_frame[0], frame[0])
        fwidth = 352
        fheight = 288
        qartist.set_UVC(mvx_gold, mvy_gold)
        # quiver(N.arange(fwidth/16) * 16 + 8, N.arange(fheight/16) * 16 + 8 , mvx_gold, mvx_gold, color='c', scale=1, units='x', angles='xy', width=fwidth*0.002)

        axes(axrf)
        imrf.set_data(previous_frame[0])

        axes(axres)
        imres.set_data(frame[0].astype(N.int16) - previous_frame[0].astype(N.int16))
        # imres = imshow((frame[0].astype(N.int16) - previous_frame[0].astype(N.int16)) / 8 + 128, cmap=cm.gray, animated=True)

        axes(axnewres)
        imnewres.set_data(res_gold)

        if options.output_dir:
            savefig(os.path.join(options.output_dir, 'frame-'+str(frameNumber)+'.png'), dpi=300)
        # if visualize:
            # show()
        # axcf.draw_artist(imcf)
        # axcf.draw_artist(qartist)
        # axrf.draw_artist(imrf)
        # axres.draw_artist(imres)
        # axnewres.draw_artist(imnewres)

        canvas.draw()

        previous_frame = frame
        return True

    # gobject.idle_add(update, frameNumber, previous_frame, video)
    def start_anim(event):
        gobject.idle_add(update, video, options)
        canvas.mpl_disconnect(start_anim.cid)

    start_anim.cid = canvas.mpl_connect('draw_event', start_anim)
    show()

if __name__ == '__main__':
    main()
