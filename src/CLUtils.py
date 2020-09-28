from skimage import transform
import numpy as np 
import cv2
from matplotlib import pyplot
from numba import jit
import thinplate as tps
def correct_labels( img, thresh=128 ) :
    """Reorder line labels by y-coordinates
    """
    line_labels = list(filter( lambda v : v>0, np.unique( img ) ))
    cy = []
    for ll in line_labels :
        ys, xs = np.nonzero( img == ll )
        if ( len(ys)<thresh ) :
            #print "reject one line"
            img[img==ll] = 0
        else :
            y = np.median( ys )
            cy.append(y)

    idx = np.argsort(cy)
    new = np.array( img, dtype='int32')
    for jj, ii in enumerate( idx ):
        nl = jj + 1
        ll = line_labels[ii]
        #print "\t", ll, "->", nl
        new[ img==ll ] = nl
    return new   

def load_image_files( file_list, prefix=None ) :
    with open( file_list ) as IN :
        filelist = [ line.strip().split(' ') for line in IN.readlines() ]
    if ( prefix is None ) :
        return filelist
    else :
        new_filelist = []
        for f1,f2 in filelist :
            new_filelist.append( [os.path.join( prefix, f1), 
                                  os.path.join( prefix, f2) ] )
        return new_filelist

class DataGenerator( object ) :
    def __init__( self, image_files, batch_size, nb_batches_per_epoch=None, mode = 'training', min_scale=.33, seed = 123567, 
                  pad = 32, target_size = (512,512), use_mirror=True ) :
        self.mode = mode
        self.batch_size = batch_size
        self.file_list = image_files
        self.min_scale = min_scale
        self.batch_idx = 0
        self.pad = pad
        self.epoch_idx = 0
        self.target_size = target_size
        self.nb_samples = len( self.file_list )
        self.nb_batches_per_epoch = min(nb_batches_per_epoch or self.nb_samples // batch_size, 500)
        self.use_mirror = use_mirror
        self._set_prng( seed )
    def _set_prng( self, seed ) :
        self.prng = np.random.RandomState( seed )
        return
    def _perspective_arugment( self, image, label, seed = None, min_scale = .75 ) :
        h, w = image.shape[:2]
        M = self._get_random_perspective_transform_matrix( h, w, min_scale )
        rimage = cv2.warpPerspective( image, M, 
                                      dsize = (w,h), 
                                      borderMode = cv2.BORDER_CONSTANT, 
                                      borderValue = 0,
                                      flags = cv2.INTER_AREA )
        rlabel = cv2.warpPerspective( label, M, 
                                      dsize = (w,h), 
                                      borderMode = cv2.BORDER_CONSTANT, 
                                      borderValue = 0,
                                      flags = cv2.INTER_NEAREST )

        return rimage, rlabel
    def _get_random_perspective_transform_matrix( self, height, width, min_scale = 0.67) :
        # Pt (x,y)                                                                          
        upper_left = (0,0)
        upper_right = (width-1,0)
        lower_left = (0,height-1)
        lower_right = (width-1,height-1)
        # get max allowed shifts
        shift_x = int( width * ( 1 - min_scale ) ) // 2
        shift_y = int( height * ( 1 - min_scale ) ) // 2
        new_upper_left = ( self.prng.randint( 0, shift_x ), self.prng.randint( 0, shift_y ) )
        new_upper_right = ( width - 1 - self.prng.randint( 0, shift_x ), self.prng.randint( 0, shift_y ) )
        new_lower_left = ( self.prng.randint( 0, shift_x ), height - 1 - self.prng.randint( 0, shift_y ) )
        new_lower_right = ( width - 1 - self.prng.randint( 0, shift_x ), height - 1 - self.prng.randint( 0, shift_y ) )
        # get transform    
        src_pts = np.row_stack( [ upper_left, upper_right, lower_right, lower_left ] ).astype( np.float32 )
        dst_pts = np.row_stack( [ new_upper_left, new_upper_right, new_lower_right, new_lower_left ] ).astype( np.float32 )
        M = cv2.getPerspectiveTransform( src_pts, dst_pts )
        return M    
    def _get_one_sample( self, idx ) :
        img_filename, gt_filename = self.file_list[idx] 
        cls = cv2.imread( gt_filename, 0 )
        img = np.float32( cls!=0 )
        # downsample if necessary
        h = img.shape[0]
        w = img.shape[1]
        #img = cv2.resize( img, self.target_size, interpolation=cv2.INTER_NEAREST )
        #cls = cv2.resize( cls, self.target_size, interpolation=cv2.INTER_NEAREST )
        #h, w = img.shape[:2]
        if (h>1280) :
            #print "resize"
            nh = 1280
            nw = int(1280./h*w)
            img = cv2.resize( img, (nw, nh), interpolation=cv2.INTER_AREA )
            cls = cv2.resize( cls, (nw, nh), interpolation=cv2.INTER_NEAREST )
            if 0:
                pyplot.figure()
                pyplot.imshow( img, cmap='gray')
                pyplot.figure()
                pyplot.imshow( cls, cmap='tab20')
                pyplot.show()
        if ( self.mode == 'training') :
            img, cls = self._perspective_arugment(img, cls, self.min_scale)
            if 0:
                pyplot.figure()
                pyplot.imshow( img, cmap='gray')
                pyplot.figure()
                pyplot.imshow( cls, cmap='tab20')
                pyplot.show()
        h, w = img.shape[:2]
        if ( h >= self.target_size[0] - 2*self.pad ) and ( w >= self.target_size[1] - 2*self.pad) :
            pass
        else :
            if ( h >= self.target_size[0] - 2*self.pad ) :
                # pad w only
                cls = np.pad( cls, ((0,0), (0, self.target_size[1]-2*self.pad-w)), mode ='constant' )
                img = np.pad( img, ((0,0), (0, self.target_size[1]-2*self.pad-w)), mode ='constant' )
                #print "case1"
            elif ( w >= self.target_size[1] - 2*self.pad) :
                cls = np.pad( cls, ((0, self.target_size[0]-2*self.pad-h), (0,0)), mode ='constant' )
                img = np.pad( img, ((0, self.target_size[0]-2*self.pad-h), (0,0)), mode ='constant' )
                #print "case2"
            else :
                cls = np.pad( cls, ((0, self.target_size[0]-2*self.pad-h), (0, self.target_size[1]-2*self.pad-w)), mode ='constant' )
                img = np.pad( img, ((0, self.target_size[0]-2*self.pad-h), (0, self.target_size[1]-2*self.pad-w)), mode ='symmetric' )
                #print "case3"
            h, w = img.shape[:2]
        i0 = self.prng.randint( 0, h-self.target_size[0]+2*self.pad+1 )            
        j0 = self.prng.randint( 0, w-self.target_size[1]+2*self.pad+1 )
        i1 = i0 + self.target_size[0]-2*self.pad
        j1 = j0 + self.target_size[1]-2*self.pad
        x = np.pad( np.float32( img[i0:i1,j0:j1] ), ((self.pad, self.pad), (self.pad, self.pad)), mode='constant' )
        y = np.pad( np.float32( cls[i0:i1,j0:j1] ), ((self.pad, self.pad), (self.pad, self.pad)), mode='constant' )
        y = cv2.dilate( y.astype('uint8'), np.ones((3,3),dtype='uint8'), iterations=1 ).astype('float32')
        val_ys = np.unique(y)
        #print val_ys
        try :
            # 1. remove top and bottom lines
            y[ y==0 ] = -1
            upper_line_label = val_ys[1]
            lower_line_label = val_ys[-1]
            x[ y==upper_line_label ] = 0
            x[ y==lower_line_label ] = 0
            y[ y==upper_line_label ] = 0
            y[ y==lower_line_label ] = 0
            #print upper_line_label, lower_line_label, "set lower line to zero"
            y[ y>0 ] -= upper_line_label
            #print 'upper', np.unique(y)
            # 2. random remove some middle line
            if ( self.mode=='training') :
                if ( self.prng.randn() > 0) :
                    line_label = self.prng.randint(1, lower_line_label-upper_line_label)
                    #print "random drop", line_label
                    x[ y==line_label ] = 0
                    y[ y==line_label ] = 0
                    y[ y>line_label ] -= 1
            #print 'middle', np.unique(y)
            # 3. random skip
            if 1 and ( self.mode == 'training') :
                if ( self.prng.randn() > 0) :
                    if ( self.prng.randn() > 0) :
                        line_label = 1
                    else :
                        line_label = y.max()
                    #print line_label, "cut half", [1, y.max()]
                    mask = y==line_label
                    x0 = np.random.randint(self.target_size[1]//4,self.target_size[1]//2)
                    #print line_label, x0, x1
                    mask[:,:x0] = False
                    x[ mask ] = 0
                    y[ mask ] = 0
            #print 'skip', np.unique(y)
            # 4. clear border
            y[:self.pad//2] = 0
            y[-self.pad//2:] = 0
            y[:,:self.pad//2] = 0
            y[:,-self.pad//2:] = 0
        except :
            pass
        y = correct_labels(y)
        #print "after correction", np.unique(y)
        x = np.expand_dims( np.expand_dims(x,axis=0), axis=-1 )
        y = np.expand_dims( np.expand_dims(y,axis=0), axis=-1 )
        return x, y
    def __getitem__( self, batch_idx ) :
        if ( self.mode == 'training' ) :
            sample_indices = self.prng.randint( 0, self.nb_samples, size=(self.batch_size,) )
        else :
            sample_indices = np.arange( self.batch_size * self.batch_idx, self.batch_size * ( self.batch_idx + 1 ) ) % self.nb_samples
            self._set_prng( batch_idx )
        bX, bXM, bY, bYM = [], [], [], []
        for idx in sample_indices :
            x, y = self._get_one_sample( idx )
            bX.append(x)
            bY.append(y)
        bX, bY = self.postprocess( bX, bY )
        if self.use_mirror :
            return bX, [bY,bY]
        else :
            return bX, bY
    def postprocess( self, bX, bY ) :
        fail = 0
        X, Y = [], []
        for x, y in zip(bX, bY) :
            if x.shape != (1, self.target_size[0], self.target_size[1], 1 ) :
                fail += 1
            elif x.shape != y.shape :
                fail += 1
            elif y.max() < 1 :
                fail += 1
            else :
                X.append(x)
                Y.append(y)
        if ( fail > 0 ) :    
            X += X[:fail]
            Y += Y[:fail]
        return [np.concatenate( X ), np.concatenate( Y )]
    def __iter__( self ) :
        return self
    def __next__( self ) :
        self.batch_idx = self.batch_idx + 1
        if ( self.batch_idx + 1 > self.nb_batches_per_epoch ) :
            self.batch_idx = 0
            self.epoch_idx += 1
        return self[ self.batch_idx ]
    
class TrainingJobs :
    def __init__(self, 
                 baseline_params,
                 search_params,
                 num_of_gpus=8) :
        #TODO
        return
class DataGenerator2( object ) :
    def __init__( self, image_files, batch_size, nb_batches_per_epoch=None, mode = 'training', min_scale=.33, seed = 123567, 
                  pad = 32, target_size = (512,512), use_mirror=True ) :
        self.mode = mode
        self.batch_size = batch_size
        self.file_list = image_files
        self.min_scale = min_scale
        self.batch_idx = 0
        self.pad = pad
        self.epoch_idx = 0
        self.target_size = target_size
        self.nb_samples = len( self.file_list )
        self.nb_batches_per_epoch = min(nb_batches_per_epoch or self.nb_samples // batch_size, 500)
        self.use_mirror = use_mirror
        self._set_prng( seed )
    def _set_prng( self, seed ) :
        self.prng = np.random.RandomState( seed )
        return
    def _perspective_arugment( self, image, label, seed = None, min_scale = .75 ) :
        h, w = image.shape[:2]
        M = self._get_random_perspective_transform_matrix( h, w, min_scale )
        rimage = cv2.warpPerspective( image, M, 
                                      dsize = (w,h), 
                                      borderMode = cv2.BORDER_CONSTANT, 
                                      borderValue = 0,
                                      flags = cv2.INTER_AREA )
        rlabel = cv2.warpPerspective( label, M, 
                                      dsize = (w,h), 
                                      borderMode = cv2.BORDER_CONSTANT, 
                                      borderValue = 0,
                                      flags = cv2.INTER_NEAREST )
       

        return rimage, rlabel
    def _get_random_perspective_transform_matrix( self, height, width, min_scale = 0.67) :
        # Pt (x,y)                                                                          
        upper_left = (0,0)
        upper_right = (width-1,0)
        lower_left = (0,height-1)
        lower_right = (width-1,height-1)
        # get max allowed shifts
        shift_x = int( width * ( 1 - min_scale ) ) // 2
        shift_y = int( height * ( 1 - min_scale ) ) // 2
        new_upper_left = ( self.prng.randint( 0, shift_x ), self.prng.randint( 0, shift_y ) )
        new_upper_right = ( width - 1 - self.prng.randint( 0, shift_x ), self.prng.randint( 0, shift_y ) )
        new_lower_left = ( self.prng.randint( 0, shift_x ), height - 1 - self.prng.randint( 0, shift_y ) )
        new_lower_right = ( width - 1 - self.prng.randint( 0, shift_x ), height - 1 - self.prng.randint( 0, shift_y ) )
        # get transform    
        src_pts = np.row_stack( [ upper_left, upper_right, lower_right, lower_left ] ).astype( np.float32 )
        dst_pts = np.row_stack( [ new_upper_left, new_upper_right, new_lower_right, new_lower_left ] ).astype( np.float32 )
        M = cv2.getPerspectiveTransform( src_pts, dst_pts )
        return M    
    def _get_one_sample( self, idx ) :
        img_filename, gt_filename = self.file_list[idx] 
        cls = cv2.imread( gt_filename, 0 )
      
        img = np.float32( cls!=0 )
        #if ( self.mode == 'training') :
            #if ( self.prng.randn() > 0) :
                

                #img, cls = self._perspective_arugment(img, cls, self.min_scale)
            #if 0:
                #pyplot.figure()
                #pyplot.imshow( img, cmap='gray')
                #pyplot.figure()
                #pyplot.imshow( cls, cmap='tab20')
                #pyplot.show()
        
    
        
       
        h = cls.shape[0]
        w = cls.shape[1]
        cof = self.target_size[0]/self.target_size[1]
        #white = np.zeros([int(w*cof),w])
        black = np.zeros([int(w*cof),w])
        if h > black.shape[0]:
            #img = cv2.resize(img,(self.target_size[1],self.target_size[0]),interpolation=cv2.INTER_NEAREST)
            cls = cv2.resize(cls,(self.target_size[1],self.target_size[0]),interpolation=cv2.INTER_NEAREST)
        else:
            #white[:h,:w] = img
            black[:h,:w] = cls
            pacent = self.target_size[0]/black.shape[0]    
            #img = cv2.resize(white,None,fx=pacent,fy=pacent,interpolation=cv2.INTER_NEAREST)
            cls = cv2.resize(black,None,fx=pacent,fy=pacent,interpolation=cv2.INTER_NEAREST)
        
        
        
        #x = img
        y = cls
      
        y =  y.astype('uint8')
        x = (y>=1).astype('float64')
       
       
              
        hl = gethl(x)
        
        
        zero_x = np.zeros([self.target_size[0],self.target_size[1]])
        zero_y = np.zeros([self.target_size[0],self.target_size[1]])
        
        zero_x[:x[hl:,:].shape[0],:] = x[hl:,:self.target_size[1]]
        zero_y[:y[hl:,:].shape[0],:] = y[hl:,:self.target_size[1]]
        y = zero_y.astype('uint8')
        x = zero_x.astype('float64')
        
        
        
    
        
        
        val_ys = np.unique(y)
        one_class = (y==1).astype('uint8')
        lh,lhx = getlh(x,one_class)
        
        top = y[:lh,:]
        below = y[lh:,:]
        below = cv2.resize(top,(below.shape[1],below.shape[0]),interpolation=cv2.INTER_NEAREST)
        below = (below>0).astype("uint8")
        y[lh:,:] = below*(np.max(y)+1)
        
        try:
    
            
            #below = cv2.resize(top,(below.shape[1],below.shape[0]),interpolation=cv2.INTER_NEAREST)
            x[lh:lh+one_class[:lhx,:].shape[0],:] = one_class[:lhx,:].astype("float64")
            
        except:
            pass
        
       
        
        #print val_ys
        '''
        try :
            # 1. remove top and bottom lines
            #y[ y==0 ] = -1
            upper_line_label = val_ys[0]
            lower_line_label = val_ys[-1]
            #x[ y==upper_line_label ] = 0
            #x[ y==lower_line_label ] = 0
            #y[ y==upper_line_label ] = 0
            #y[ y==lower_line_label ] = 0
            #print upper_line_label, lower_line_label, "set lower line to zero"
            #y[ y>0 ] -= upper_line_label
            #print 'upper', np.unique(y)
            # 2. random remove some middle line
            if ( self.mode=='training') :
                if ( self.prng.randn() > 0) :
                    line_label = self.prng.randint(1, lower_line_label-upper_line_label)
                    #print "random drop", line_label
                    x[ y==line_label ] = 0
                    y[ y==line_label ] = 0
                    y[ y>line_label ] -= 1
                   
            #print 'middle', np.unique(y)
            # 3. random skip
            #if 1 and ( self.mode == 'training') :
                #if ( self.prng.randn() > 0) :
                    #if ( self.prng.randn() > 0) :
                        #line_label = 1
                    #else :
                        #line_label = y.max()
                    #print line_label, "cut half", [1, y.max()]
                    #mask = y==line_label
                    #x0 = np.random.randint(self.target_size[1]//4,self.target_size[1]//2)
                    #print line_label, x0, x1
                    #mask[:,:x0] = False
                    #x[ mask ] = 0
                    #y[ mask ] = 0
            #print 'skip', np.unique(y)
            # 4. clear border
            #y[:self.pad//2] = 0
            #y[-self.pad//2:] = 0
            #y[:,:self.pad//2] = 0
            #y[:,-self.pad//2:] = 0
        except :
            pass
        '''
        y = correct_labels(y)
        #print "after correction", np.unique(y)
        x = np.expand_dims( np.expand_dims(x,axis=0), axis=-1 )
        y = np.expand_dims( np.expand_dims(y,axis=0), axis=-1 )
        return x, y
    def __getitem__( self, batch_idx ) :
        if ( self.mode == 'training' ) :
            sample_indices = self.prng.randint( 0, self.nb_samples, size=(self.batch_size,) )
        else :
            sample_indices = np.arange( self.batch_size * self.batch_idx, self.batch_size * ( self.batch_idx + 1 ) ) % self.nb_samples
            self._set_prng( batch_idx )
        bX, bXM, bY, bYM = [], [], [], []
        for idx in sample_indices :
            x, y = self._get_one_sample( idx )
            bX.append(x)
            bY.append(y)
        bX, bY = self.postprocess( bX, bY )
        if self.use_mirror :
            return bX, [bY,bY]
        else :
            return bX, bY
    def postprocess( self, bX, bY ) :
        fail = 0
        X, Y = [], []
        for x, y in zip(bX, bY) :
            if x.shape != (1, self.target_size[0], self.target_size[1], 1 ) :
                fail += 1
            elif x.shape != y.shape :
                fail += 1
            elif y.max() < 1 :
                fail += 1
            else :
                X.append(x)
                Y.append(y)
        if ( fail > 0 ) :    
            X += X[:fail]
            Y += Y[:fail]
        return [np.concatenate( X ), np.concatenate( Y )]
    def __iter__( self ) :
        return self
    def __next__( self ) :
        self.batch_idx =  self.batch_idx + 1
        if ( self.batch_idx + 1 > self.nb_batches_per_epoch ) :
            self.batch_idx = 0
            self.epoch_idx += 1
        return self[ self.batch_idx ]
def load_image_files( file_list, prefix=None ) :
    with open( file_list ) as IN :
        filelist = [ line.strip().split(' ') for line in IN.readlines() ]
    if ( prefix is None ) :
        return filelist
    else :
        new_filelist = []
        for f1,f2 in filelist :
            new_filelist.append( [os.path.join( prefix, f1), 
                                  os.path.join( prefix, f2) ] )
        return new_filelist

class DataGenerator3( object ) :
    def __init__( self, image_files, batch_size, nb_batches_per_epoch=None, mode = 'training', min_scale=.33, seed = 123567, 
                  pad = 32, target_size = (512,512), use_mirror=True ) :
        self.mode = mode
        self.batch_size = batch_size
        self.file_list = image_files
        self.min_scale = min_scale
        self.batch_idx = 0
        self.pad = pad
        self.epoch_idx = 0
        self.target_size = target_size
        self.nb_samples = len( self.file_list )
        self.nb_batches_per_epoch = min(nb_batches_per_epoch or self.nb_samples // batch_size, 500)
        self.use_mirror = use_mirror
        self._set_prng( seed )
    def _set_prng( self, seed ) :
        self.prng = np.random.RandomState( seed )
        return
    def _perspective_arugment( self, image, label, seed = None, min_scale = .75 ) :
        h, w = image.shape[:2]
        M = self._get_random_perspective_transform_matrix( h, w, min_scale )
        rimage = cv2.warpPerspective( image, M, 
                                      dsize = (w,h), 
                                      borderMode = cv2.BORDER_CONSTANT, 
                                      borderValue = 0,
                                      flags = cv2.INTER_NEAREST )
        rlabel = cv2.warpPerspective( label, M, 
                                      dsize = (w,h), 
                                      borderMode = cv2.BORDER_CONSTANT, 
                                      borderValue = 0,
                                      flags = cv2.INTER_NEAREST )

        return rimage, rlabel
    def warp_image_cv(self,img, c_src, c_dst, dshape=None):
        dshape = dshape or img.shape
        theta = tps.tps_theta_from_points(c_src, c_dst, reduced=True)
        grid = tps.tps_grid(theta, c_dst, dshape)
        mapx, mapy = tps.tps_grid_to_remap(grid, img.shape)
        return cv2.remap(img, mapx, mapy, cv2.INTER_NEAREST)
    def thinplat(self,img,gt,h,w):
        dev = 0.07
        csl = 0.35 +  float(np.random.uniform(-dev,dev,1))
        cst = 0.35 +  float(np.random.uniform(-dev,dev,1))
        csb = 0.65 +  float(np.random.uniform(-dev,dev,1))
        csr = 0.65 +  float(np.random.uniform(-dev,dev,1))
        cdl = 0.35 +  float(np.random.uniform(-dev,dev,1))
        cdt = 0.35 +  float(np.random.uniform(-dev,dev,1))
        cdb = 0.65 +  float(np.random.uniform(-dev,dev,1))
        cdr = 0.65 +  float(np.random.uniform(-dev,dev,1))

        dev = 0.2
        lt = float(np.random.uniform(0,dev,1))
        rt = float(np.random.uniform(0,dev,1))
        lm = float(np.random.uniform(0,dev,1))
        rm = float(np.random.uniform(0,dev,1))
        lm2 = float(np.random.uniform(0,dev,1))
        rm2 = float(np.random.uniform(0,dev,1))
        lb = float(np.random.uniform(0,dev,1))
        rb = float(np.random.uniform(0,dev,1))
        c_src = np.array([
            [lt, rt],
            [lm, rm],
            [lm2, rm2],
            [lb, lb],
            [csl, cst],
            [csb, csr],
        ])

        c_dst = np.array([
            [lt, rt],
            [lm, rm],
            [lm2, rm2],
            [lb, lb],
            [cdl, cdt],
            [cdb, cdr],
        ])

        img = self.warp_image_cv(img, c_src, c_dst, dshape=(h,w))
        gt = self.warp_image_cv(gt, c_src, c_dst, dshape=(h,w))
        return img,gt

    def _get_random_perspective_transform_matrix( self, height, width, min_scale = 0.67) :
        # Pt (x,y)                                                                          
        upper_left = (0,0)
        upper_right = (width-1,0)
        lower_left = (0,height-1)
        lower_right = (width-1,height-1)
        # get max allowed shifts
        shift_x = int( width * ( 1 - min_scale ) ) // 2
        shift_y = int( height * ( 1 - min_scale ) ) // 2
        new_upper_left = ( self.prng.randint( 0, shift_x ), self.prng.randint( 0, shift_y ) )
        new_upper_right = ( width - 1 - self.prng.randint( 0, shift_x ), self.prng.randint( 0, shift_y ) )
        new_lower_left = ( self.prng.randint( 0, shift_x ), height - 1 - self.prng.randint( 0, shift_y ) )
        new_lower_right = ( width - 1 - self.prng.randint( 0, shift_x ), height - 1 - self.prng.randint( 0, shift_y ) )
        # get transform    
        src_pts = np.row_stack( [ upper_left, upper_right, lower_right, lower_left ] ).astype( np.float32 )
        dst_pts = np.row_stack( [ new_upper_left, new_upper_right, new_lower_right, new_lower_left ] ).astype( np.float32 )
        M = cv2.getPerspectiveTransform( src_pts, dst_pts )
        return M    
    def _get_one_sample( self, idx ) :
        img_filename, gt_filename = self.file_list[idx] 
        cls = cv2.imread( gt_filename, 0 )
        img = np.float32( cls!=0 )
        h = img.shape[0]
        w = img.shape[1]
        cof = self.target_size[0]/self.target_size[1]
        white = np.zeros([int(w*cof),w])
        black = np.zeros([int(w*cof),w])
        if h > white.shape[0]:
            img = cv2.resize(img,(self.target_size[1],self.target_size[0]),interpolation=cv2.INTER_NEAREST)
            cls = cv2.resize(cls,(self.target_size[1],self.target_size[0]),interpolation=cv2.INTER_NEAREST)
        else:
            white[:h,:w] = img
            black[:h,:w] = cls
            pacent = self.target_size[0]/white.shape[0]    
            img = cv2.resize(white,None,fx=pacent,fy=pacent,interpolation=cv2.INTER_NEAREST)
            cls = cv2.resize(black,None,fx=pacent,fy=pacent,interpolation=cv2.INTER_NEAREST)
        
        if ( self.mode == 'training'):
            if ( self.prng.randn() > 0) :
                if ( self.prng.randn() > 0) :
                    img, cls = self.thinplat(img, cls, self.target_size[0],self.target_size[1])
                else:
                    img, cls = self._perspective_arugment(img, cls, self.min_scale)
        #if ( self.mode == 'training') :
            #if ( self.prng.randn() > 0) :
                #img, cls = self._perspective_arugment(img, cls, self.min_scale)
            #if 0:
                #pyplot.figure()
                #pyplot.imshow( img, cmap='gray')
                #pyplot.figure()
                #pyplot.imshow( cls, cmap='tab20')
                #pyplot.show()
        x = img
        y = cls
        y =  y.astype('uint8')
        if ( self.mode != 'testing') :
            
            #img, cls = self.thinplat(img, cls, self.target_size[0],self.target_size[1])
            hl = gethl(x)
            lh = getlh(x)
       
            top = cv2.resize(y[hl:lh,:],(y[:hl,:].shape[1],y[:hl,:].shape[0]),interpolation=cv2.INTER_NEAREST)
            top = (top>0).astype("uint8")
            y[:hl] = top
        
        
       
      
        
        val_ys = np.unique(y)
        
        #print val_ys
        try :
            # 1. remove top and bottom lines
            #y[ y==0 ] = -1
            upper_line_label = val_ys[1]
            lower_line_label = val_ys[-1]
            #x[ y==upper_line_label ] = 0
            #x[ y==lower_line_label ] = 0
            #y[ y==upper_line_label ] = 0
            #y[ y==lower_line_label ] = 0
            #print upper_line_label, lower_line_label, "set lower line to zero"
            #y[ y>0 ] -= upper_line_label
            #print 'upper', np.unique(y)
            # 2. random remove some middle line
            if ( self.mode=='training') :
                if ( self.prng.randn() > 0) :
                    line_label = self.prng.randint(1, lower_line_label-upper_line_label)
                    #print "random drop", line_label
                    x[ y==line_label ] = 0
                    y[ y==line_label ] = 0
                    y[ y>line_label ] -= 1
            #print 'middle', np.unique(y)

            
            #print 'skip', np.unique(y)
            # 4. clear border
            #y[:self.pad//2] = 0
            #y[-self.pad//2:] = 0
            #y[:,:self.pad//2] = 0
            #y[:,-self.pad//2:] = 0
        except :
            pass
        
        y = correct_labels(y)
        #print "after correction", np.unique(y)
        x = np.expand_dims( np.expand_dims(x,axis=0), axis=-1 )
        y = np.expand_dims( np.expand_dims(y,axis=0), axis=-1 )
        return x, y
    def __getitem__( self, batch_idx ) :
        if ( self.mode == 'training' ) :
            sample_indices = self.prng.randint( 0, self.nb_samples, size=(self.batch_size,) )
        else :
            sample_indices = np.arange( self.batch_size * self.batch_idx, self.batch_size * ( self.batch_idx + 1 ) ) % self.nb_samples
            self._set_prng( batch_idx )
        bX, bXM, bY, bYM = [], [], [], []
        for idx in sample_indices :
            x, y = self._get_one_sample( idx )
            bX.append(x)
            bY.append(y)
        bX, bY = self.postprocess( bX, bY )
        if self.use_mirror :
            return bX, [bY,bY]
        else :
            return bX, bY
    def postprocess( self, bX, bY ) :
        fail = 0
        X, Y = [], []
        for x, y in zip(bX, bY) :
            if x.shape != (1, self.target_size[0], self.target_size[1], 1 ) :
                fail += 1
            elif x.shape != y.shape :
                fail += 1
            elif y.max() < 1 :
                fail += 1
            else :
                X.append(x)
                Y.append(y)
        if ( fail > 0 ) :    
            X += X[:fail]
            Y += Y[:fail]
        return [np.concatenate( X ), np.concatenate( Y )]
    def __iter__( self ) :
        return self
    def __next__( self ) :
        self.batch_idx =  self.batch_idx + 1
        if ( self.batch_idx + 1 > self.nb_batches_per_epoch ) :
            self.batch_idx = 0
            self.epoch_idx += 1
        return self[ self.batch_idx ]
def load_image_files( file_list, prefix=None ) :
    with open( file_list ) as IN :
        filelist = [ line.strip().split(' ') for line in IN.readlines() ]
    if ( prefix is None ) :
        return filelist
    else :
        new_filelist = []
        for f1,f2 in filelist :
            new_filelist.append( [os.path.join( prefix, f1), 
                                  os.path.join( prefix, f2) ] )
        return new_filelist

@jit
def gethl(img):
    for i in range(img.shape[0]-1,0,-1):
         for j in range(img.shape[1]-1,0,-1):
            if img[i,j] > 0:
                 hl = i
    return hl
@jit
def getlh(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j] > 0:
                lh = i
    return lh