import numpy as np
from core.leras import nn
tf = nn.tf
    
patch_discriminator_kernels = \
    { 1  : (512, [ [1,1] ]),
      2  : (512, [ [2,1] ]),
      3  : (512, [ [2,1], [2,1] ]),
      4  : (512, [ [2,2], [2,2] ]),
      5  : (512, [ [3,2], [2,2] ]),
      6  : (512, [ [4,2], [2,2] ]),
      7  : (512, [ [3,2], [3,2] ]),
      8  : (512, [ [4,2], [3,2] ]),
      9  : (512, [ [3,2], [4,2] ]),
      10 : (512, [ [4,2], [4,2] ]),      
      11 : (512, [ [3,2], [3,2], [2,1] ]),
      12 : (512, [ [4,2], [3,2], [2,1] ]),
      13 : (512, [ [3,2], [4,2], [2,1] ]),
      14 : (512, [ [4,2], [4,2], [2,1] ]),
      15 : (512, [ [3,2], [3,2], [3,1] ]),
      16 : (512, [ [4,2], [3,2], [3,1] ]),
      17 : (512, [ [3,2], [4,2], [3,1] ]),
      18 : (512, [ [4,2], [4,2], [3,1] ]),      
      19 : (512, [ [3,2], [3,2], [4,1] ]),
      20 : (512, [ [4,2], [3,2], [4,1] ]),
      21 : (512, [ [3,2], [4,2], [4,1] ]),
      22 : (512, [ [4,2], [4,2], [4,1] ]),      
      23 : (256, [ [3,2], [3,2], [3,2], [2,1] ]),
      24 : (256, [ [4,2], [3,2], [3,2], [2,1] ]),
      25 : (256, [ [3,2], [4,2], [3,2], [2,1] ]),
      26 : (256, [ [4,2], [4,2], [3,2], [2,1] ]),      
      27 : (256, [ [3,2], [4,2], [4,2], [2,1] ]),      
      28 : (128, [ [4,2], [3,2], [4,2], [2,1] ]),
      29 : (256, [ [3,2], [4,2], [4,2], [2,1] ]),
      30 : (256, [ [4,2], [4,2], [4,2], [2,1] ]),      
      31 : (256, [ [3,2], [3,2], [3,2], [3,1] ]),
      32 : (256, [ [4,2], [3,2], [3,2], [3,1] ]),
      33 : (256, [ [3,2], [4,2], [3,2], [3,1] ]),
      34 : (256, [ [4,2], [4,2], [3,2], [3,1] ]),      
      35 : (256, [ [3,2], [4,2], [4,2], [3,1] ]),      
      36 : (256, [ [4,2], [3,2], [4,2], [3,1] ]),
      37 : (256, [ [3,2], [4,2], [4,2], [3,1] ]),
      38 : (256, [ [4,2], [4,2], [4,2], [3,1] ]),      
      39 : (256, [ [3,2], [3,2], [3,2], [4,1] ]),
      40 : (256, [ [4,2], [3,2], [3,2], [4,1] ]),
      41 : (256, [ [3,2], [4,2], [3,2], [4,1] ]),
      42 : (256, [ [4,2], [4,2], [3,2], [4,1] ]),      
      43 : (256, [ [3,2], [4,2], [4,2], [4,1] ]),      
      44 : (256, [ [4,2], [3,2], [4,2], [4,1] ]),
      45 : (256, [ [3,2], [4,2], [4,2], [4,1] ]),
      46 : (256, [ [4,2], [4,2], [4,2], [4,1] ]),
    }


class PatchDiscriminator(nn.ModelBase):
    def on_build(self, patch_size, in_ch, base_ch=None, conv_kernel_initializer=None):            
        suggested_base_ch, kernels_strides = patch_discriminator_kernels[patch_size]
        
        if base_ch is None:
            base_ch = suggested_base_ch
        
        prev_ch = in_ch
        self.convs = []
        for i, (kernel_size, strides) in enumerate(kernels_strides):                
            cur_ch = base_ch * min( (2**i), 8 )
            
            self.convs.append ( nn.Conv2D( prev_ch, cur_ch, kernel_size=kernel_size, strides=strides, padding='SAME', kernel_initializer=conv_kernel_initializer) )
            prev_ch = cur_ch

        self.out_conv =  nn.Conv2D( prev_ch, 1, kernel_size=1, padding='VALID', kernel_initializer=conv_kernel_initializer)

    def forward(self, x):
        for conv in self.convs:
            x = tf.nn.leaky_relu( conv(x), 0.1 )
        return self.out_conv(x)

nn.PatchDiscriminator = PatchDiscriminator

class UNetPatchDiscriminator(nn.ModelBase):
    def calc_receptive_field_size(self, layers):
        rf = 0
        ts = 1
        for i, (k, s) in enumerate(layers):
            if i == 0:
                rf = k
            else:
                rf += (k-1)*ts
            ts *= s
        return rf
        
    def find_archi(self, target_patch_size, max_layers=6):
        s = {}
        for layers_count in range(1,max_layers+1):
            val = 1 << (layers_count-1)
            while True:
                val -= 1
                
                layers = []
                sum_st = 0
                for i in range(layers_count-1):
                    st = 1 + (1 if val & (1 << i) !=0 else 0 )                
                    layers.append ( [3, st ])
                    sum_st += st
                layers.append ( [3, 2])
                sum_st += 2                               
                    
                rf = self.calc_receptive_field_size(layers)
                
                s_rf = s.get(rf, None)
                if s_rf is None:
                    s[rf] = (layers_count, sum_st, layers)
                else:
                    if layers_count < s_rf[0] or \
                    ( layers_count == s_rf[0] and sum_st > s_rf[1] ):
                        s[rf] = (layers_count, sum_st, layers)
                    
                if val == 0:
                    break
                
        x = sorted(list(s.keys()))
        q=x[np.abs(np.array(x)-target_patch_size).argmin()]
        return s[q][2]
        
    def on_build(self, patch_size, in_ch):     
        class ConvBlock(nn.ModelBase):
            def on_build(self, in_ch, out_ch, kernel_size, strides):              
                self.conv = nn.Conv2D (in_ch, out_ch, kernel_size=kernel_size, strides=strides, padding='SAME')
                self.frn = nn.FRNorm2D(out_ch)
                self.tlu = nn.TLU(out_ch)

            def forward(self, x):                
                x = self.conv(x)
                x = self.frn(x)
                x = self.tlu(x)
                return x

        class UpConvBlock(nn.ModelBase):
            def on_build(self, in_ch, out_ch, kernel_size, strides):              
                self.conv = nn.Conv2DTranspose (in_ch, out_ch, kernel_size=kernel_size, strides=strides, padding='SAME')
                self.frn = nn.FRNorm2D(out_ch)
                self.tlu = nn.TLU(out_ch)

            def forward(self, x):
                x = self.conv(x)
                x = self.frn(x)
                x = self.tlu(x)
                return x
                
        class ResidualBlock2(nn.ModelBase):
            def on_build(self, ch, kernel_size=3 ):
                self.conv1 = nn.Conv2D( ch, ch, kernel_size=kernel_size, padding='SAME')
                self.frn1 = nn.FRNorm2D(ch)
                self.tlu1 = nn.TLU(ch)
                self.conv2 = nn.Conv2D( ch, ch, kernel_size=kernel_size, padding='SAME')
                self.frn2 = nn.FRNorm2D(ch)
                self.tlu2 = nn.TLU(ch)
                
            def forward(self, inp):
                x = self.conv1(inp)
                x = self.frn1(x)
                x = self.tlu1(x)
                x = self.conv2(x)
                x = self.frn2(x)
                x = self.tlu2(inp + x)
                return x
                                   
        class ResidualBlock(nn.ModelBase):
            def on_build(self, ch, kernel_size=3 ):
                self.conv1 = nn.Conv2D( ch, ch, kernel_size=kernel_size, padding='SAME')
                self.conv2 = nn.Conv2D( ch, ch, kernel_size=kernel_size, padding='SAME')

            def forward(self, inp):
                x = self.conv1(inp)
                x = tf.nn.leaky_relu(x, 0.2)
                x = self.conv2(x)
                x = tf.nn.leaky_relu(inp + x, 0.2)
                return x
                
        prev_ch = in_ch
        self.convs = []
        self.res = []
        self.upconvs = []
        self.upres = []
        layers = self.find_archi(patch_size)
        #base_ch = {1:384, 2:256, 3:256, 4:192, 5:128, 6:64}[len(layers)]        
        base_ch = 16#np.clip( 256-32*len(layers), 16, 512 )
        
        level_chs = { i-1:v for i,v in enumerate([ min( base_ch * (2**i), 512 ) for i in range(len(layers)+1)]) }
        
        self.in_conv = nn.Conv2D( in_ch, level_chs[-1], kernel_size=1, padding='VALID')

        for i, (kernel_size, strides) in enumerate(layers):         
            self.convs.append ( nn.Conv2D( level_chs[i-1], level_chs[i], kernel_size=kernel_size, strides=strides, padding='SAME') )
            self.res.append ( ResidualBlock(level_chs[i]) )
            self.upconvs.insert (0, nn.Conv2DTranspose( level_chs[i]*2, level_chs[i-1], kernel_size=kernel_size, strides=strides, padding='SAME') )
            
            #self.upconvs.insert (0, nn.Conv2DTranspose( level_chs[i]*(2 if i != len(layers)-1 else 1), level_chs[i-1], kernel_size=kernel_size, strides=strides, padding='SAME') )
            
            self.upres.insert (0, ResidualBlock(level_chs[i-1]) )
        self.out_conv = nn.Conv2D( level_chs[-1], 1, kernel_size=1, padding='VALID')

        self.center_out  =  nn.Conv2D( level_chs[len(layers)-1], 1, kernel_size=1, padding='VALID')
        self.center_conv =  nn.Conv2D( level_chs[len(layers)-1], level_chs[len(layers)-1], kernel_size=1, padding='VALID')


    def forward(self, x):        
        x = tf.nn.leaky_relu( self.in_conv(x), 0.1 )
        
        encs = []
        for conv, res in zip(self.convs, self.res):
            x = tf.nn.leaky_relu( conv(x), 0.1 )
            x = res(x)
            encs.insert(0, x)
            
        center_out, x = self.center_out(x), self.center_conv(x)

        for i, (upconv, enc, upres) in enumerate(zip(self.upconvs, encs, self.upres)):
            x = tf.concat( [enc, x], axis=nn.conv2d_ch_axis)
            x = tf.nn.leaky_relu(upconv(x), 0.1 )      
            x = upres(x)
            
        x = self.out_conv(x)
        

        return center_out, x
        
    def forward2(self, x):        
        x = tf.nn.leaky_relu( self.in_conv(x), 0.1 )
        
        encs = []
        for conv, res in zip(self.convs, self.res):
            x = tf.nn.leaky_relu( conv(x), 0.1 )
            x = res(x)
            encs.insert(0, x)
            
        center_out, x = self.center_out(x), self.center_conv(x)

        for i, (upconv, enc, upres) in enumerate(zip(self.upconvs, encs, self.upres)):
            x = tf.nn.leaky_relu(upconv(x), 0.1 )                 
            x = tf.concat( [enc, x], axis=nn.conv2d_ch_axis)
            x = upres(x)
            
        x = self.out_conv(x)
        

        return center_out, x
        
nn.UNetPatchDiscriminator = UNetPatchDiscriminator

"""

class UNetPatchDiscriminator(nn.ModelBase):
    def calc_receptive_field_size(self, layers):
        rf = 0
        ts = 1
        for i, (k, s) in enumerate(layers):
            if i == 0:
                rf = k
            else:
                rf += (k-1)*ts
            ts *= s
        return rf
        
    def find_archi(self, target_patch_size, max_layers=6):
        s = {}
        for layers_count in range(1,max_layers+1):
            val = 1 << (layers_count-1)
            while True:
                val -= 1
                
                layers = []
                sum_st = 0
                for i in range(layers_count-1):
                    st = 1 + (1 if val & (1 << i) !=0 else 0 )                
                    layers.append ( [3, st ])
                    sum_st += st
                layers.append ( [3, 2])
                sum_st += 2                               
                    
                rf = self.calc_receptive_field_size(layers)
                print(layers)
                print(rf)
                
                s_rf = s.get(rf, None)
                if s_rf is None:
                    s[rf] = (layers_count, sum_st, layers)
                else:
                    if layers_count < s_rf[0] or \
                    ( layers_count == s_rf[0] and sum_st > s_rf[1] ):
                        s[rf] = (layers_count, sum_st, layers)
                    
                if val == 0:
                    break
                
        x = sorted(list(s.keys()))
        q=x[np.abs(np.array(x)-target_patch_size).argmin()]
        return s[q][2]
        
        
    def on_build(self, patch_size, in_ch, base_ch=None, conv_kernel_initializer=None):            
        suggested_base_ch, kernels_strides = patch_discriminator_kernels[patch_size]
        
        if base_ch is None:
            base_ch = suggested_base_ch
        
        prev_ch = in_ch
        self.convs = []
        self.upconvs = []
        
        level_chs = { i:v for i,v in enumerate([ base_ch * min( (2**i), 8 ) for i in range(len(kernels_strides))]) }
        level_chs[-1] = in_ch
        

        for i, (kernel_size, strides) in enumerate(kernels_strides):         
            self.convs.append ( nn.Conv2D( level_chs[i-1], level_chs[i], kernel_size=kernel_size, strides=strides, padding='FORCE_SAME', kernel_initializer=conv_kernel_initializer) )
            
            self.upconvs.insert (0, nn.Conv2DTranspose( level_chs[i]*2, level_chs[i-1] if i != 0 else 1, kernel_size=kernel_size, strides=strides, padding='SAME', kernel_initializer=conv_kernel_initializer) )


        self.center_out  =  nn.Conv2D( level_chs[len(kernels_strides)-1], 1, kernel_size=1, padding='VALID', kernel_initializer=conv_kernel_initializer)
        self.center_conv =  nn.Conv2D( level_chs[len(kernels_strides)-1], level_chs[len(kernels_strides)-1], kernel_size=1, padding='VALID', kernel_initializer=conv_kernel_initializer)


        #self.conv = nn.Conv2DTranspose (in_ch, out_ch, kernel_size=3, padding='SAME')
        
    def forward(self, x):
        encs = []
        for conv in self.convs:
            x = tf.nn.leaky_relu( conv(x), 0.1 )
            encs.insert(0, x)
            
        center_out = self.center_out(x)
        
        x = self.center_conv(x)

        for i, (upconv, enc) in enumerate(zip(self.upconvs, encs)):
            
        
            x = tf.concat( [enc, x], axis=nn.conv2d_ch_axis)

            x = upconv(x)         
            if i < len(encs)-1:
                x = tf.nn.leaky_relu(x, 0.1 )         
                    
        #import code
        #code.interact(local=dict(globals(), **locals()))

        return center_out, x
    
nn.UNetPatchDiscriminator = UNetPatchDiscriminator
"""