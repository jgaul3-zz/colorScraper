colorScraper.py iterates through a given comic on tapas.io,
downloads each page from each chapter, performs k-means clustering on
that page, and saves the resulting colors in an image file.

The user parameters to edit are initialId for the first chapter
of the desired comic, the number of dominant colors to pull out,
the width of each column of color in the desired image in pixels,
and the thickness of each page's bar of pixels.
Adding colors to be blacklisted can help remove signal from noise
if each page has a color that is not desired to quantify.


colorScraperParallelClearer.py and colorScraperParallelEfficient.py use
multiprocessing.dummy's Pool to download and process the pages in parallel,
resulting in a 50% speedup using only my computer's two cores - using more may
be even better for you. ...Clearer.py has more helpful console messages, but
...Efficient.py runs quicker. These should be considered the primary release 
for this repo.

Author: Jonathan Gaul
