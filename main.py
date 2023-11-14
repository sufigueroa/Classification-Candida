import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from utils import *

PATH = 'images/Low concentration 1.tiff'
# PATH = 'images/Medium concentration 1.tiff'
# PATH = 'images/High concentration 1.tiff'

# Leer tiff y convertirlo en np.array
im = read_tiff(PATH)

im = isotropic_interpolation(im)
# im = im[:60]
segmentate_matrix(im)


############ HIFAS
# rate:  10.596363197837324
# min eccentricity:  0.9049535265890867
# mean eccentricity:  0.9374774217479481
# max eccentricity:  0.9498224685260896

# rate:  12.990635881688565
# min eccentricity:  0.6109463802079399
# mean eccentricity:  0.8468798621279702
# max eccentricity:  0.9007730304234722

# rate:  8.682242203439698
# min eccentricity:  0.5978528318697157
# mean eccentricity:  0.911329861963979
# max eccentricity:  0.99552170964637

# rate:  11.956513894985635
# min eccentricity:  0.939583099934476
# mean eccentricity:  0.949365415541194
# max eccentricity:  0.9653083448624742

# rate:  11.025211291441394
# min eccentricity:  0.9508111719254714
# mean eccentricity:  0.95782681236974
# max eccentricity:  0.9770333173300144

# rate:  10.116034883157797
# min eccentricity:  0.9964785634651389
# mean eccentricity:  0.9971407088230795
# max eccentricity:  0.9981638676939986

# rate:  11.256589305681105
# min eccentricity:  0.9110691854518066
# mean eccentricity:  0.9626300198932058
# max eccentricity:  0.9771084938220613

# rate:  12.01681284843585
# min eccentricity:  0.9780440578426302
# mean eccentricity:  0.9855101103016487
# max eccentricity:  0.9897567840382432

# rate:  11.794066943011298
# min eccentricity:  0.8811918045090159
# mean eccentricity:  0.9202748106595721
# max eccentricity:  0.9438851380482848

# rate:  8.884582310566541
# min eccentricity:  0.5520251774624718
# mean eccentricity:  0.9025847441280445
# max eccentricity:  0.9968000957790123

# rate:  14.294460355563748
# min eccentricity:  0.813254941702829
# mean eccentricity:  0.8822808506709329
# max eccentricity:  0.9108583695282572

# rate:  12.182176156232492
# min eccentricity:  0.7937877578482516
# mean eccentricity:  0.9253914375797746
# max eccentricity:  0.9521932938076011

# rate:  9.018468835182807
# min eccentricity:  0.9711668036704245
# mean eccentricity:  0.9770228578400002
# max eccentricity:  0.9922420417997295

# Hay 17 esporas y 13 esporas con hifas





# rate:  8.639740314102253
# min eccentricity:  0.4872028656737793
# mean eccentricity:  0.722947054981614
# max eccentricity:  0.8803834955881326
# rate:  9.837603430450372
# min eccentricity:  0.5004441676298019
# mean eccentricity:  0.7857642090553291
# max eccentricity:  0.8371755802104941
# rate:  10.191530231936337
# min eccentricity:  0.3684818656880347
# mean eccentricity:  0.565768698496511
# max eccentricity:  0.7399840892614575
# rate:  10.99654478849593
# min eccentricity:  0.38810889498704226
# mean eccentricity:  0.7838547511296452
# max eccentricity:  0.8390313836151321
# rate:  11.54554515793806
# min eccentricity:  0.5224770462353981
# mean eccentricity:  0.655371179541839
# max eccentricity:  0.7196109070607246
# rate:  11.780540971328296
# min eccentricity:  0.25758500260016104
# mean eccentricity:  0.4050701318172381
# max eccentricity:  0.5182579021818354
# rate:  8.728171141753572
# min eccentricity:  0.22104065675334902
# mean eccentricity:  0.509959395229744
# max eccentricity:  0.689063225788505
# rate:  11.466136954623703
# min eccentricity:  0.4449687004541415
# mean eccentricity:  0.5148962017369211
# max eccentricity:  0.6884445492579143
# rate:  8.87094852682822
# min eccentricity:  0.0
# mean eccentricity:  0.8035902904173026
# max eccentricity:  0.9648806580046635
# rate:  9.36617113385833
# min eccentricity:  0.24768254328570982
# mean eccentricity:  0.465114037999758
# max eccentricity:  0.8972890050806895
# rate:  11.762403139750958
# min eccentricity:  0.3956154616706449
# mean eccentricity:  0.7774991713087205
# max eccentricity:  0.8691198280626342
# rate:  8.209887520458544
# min eccentricity:  0.2602471626078519
# mean eccentricity:  0.624238382747507
# max eccentricity:  0.7896660798526911
# rate:  9.888326684176441
# min eccentricity:  0.3584210108667868
# mean eccentricity:  0.621601562752281
# max eccentricity:  0.7068598723481073
# rate:  9.218546034017145
# min eccentricity:  0.40611011620115123
# mean eccentricity:  0.5899945275141233
# max eccentricity:  0.8268903644422871
# rate:  4.993760812949322
# min eccentricity:  0.0
# mean eccentricity:  0.7743724816172559
# max eccentricity:  0.9571124702889098
# rate:  8.761329097836253
# min eccentricity:  0.0
# mean eccentricity:  0.42359972644405564
# max eccentricity:  0.9794878257142292
# rate:  2.8480491128904015
# min eccentricity:  0.38188130791298686
# mean eccentricity:  0.6774978666831016
# max eccentricity:  0.7865820139143213