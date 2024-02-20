import sys
sys.path.append('../')
from pycore.tikzeng import *
from pycore.blocks  import *

# defined your arch
arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),
    # input
    to_input('../india.png'),

    #DENOISE
    to_Conv("CNN_denoise_Conv0", 145, 128, offset="(0,0,0)", to="(0,0,0)", height=32, depth=32, width=2),
    to_Pool("BN0", offset="(0,0,0)", to="(CNN_denoise_Conv0-east)", height=32, depth=32, width=1),
    to_act("RELU0", offset="(0,0,0)", to="(BN0-east)", height=32, depth=32, width=1),

    to_Conv("CNN_denoise_Conv1", 145, 128, offset="(1,0,0)", to="(RELU0-east)", height=32, depth=32, width=2),
    to_Pool("BN1", offset="(0,0,0)", to="(CNN_denoise_Conv1-east)", height=32, depth=32, width=1),
    to_act("RELU1", offset="(0,0,0)", to="(BN1-east)", height=32, depth=32,width=1),
    to_connection("RELU0", "CNN_denoise_Conv1"),

    #CNN_Branch
    #depth-wise1
    *block_deepconv(name="depth_conv0", botton="RELU1", top='point_conv0', s_filer=145, n_filer=128, offset="(2,0,0)",
                  size=(32, 32, 1.0), opacity=0.1),
    to_POINT("point_conv0", 128, offset="(1.0,0,0)", to="(depth_conv0-east)", ),
    to_connection("depth_conv0", "point_conv0"),

    to_act("RELU2", offset="(1.0,0,0)", to="(point_conv0-east)", height=32, depth=32,width=1),
    to_Pool("BN2", offset="(0,0,0)", to="(RELU2-east)", height=32, depth=32, width=1),
    to_connection("point_conv0", "BN2"),

    *block_deepconv(name="depth_conv1", botton="BN2", top='point_conv1', s_filer=145, n_filer=128, offset="(2,0,0)",
                  size=(32, 32, 1.0), opacity=0.1),
    to_POINT("point_conv1", 128, offset="(1.0,0,0)", to="(depth_conv1-east)"),
    to_connection("depth_conv1", "point_conv1"),

    to_act("RELU3", offset="(1.0,0,0)", to="(point_conv1-east)", height=32, depth=32, width=1),
    to_Pool("BN3", offset="(0,0,0)", to="(RELU3-east)", height=32, depth=32, width=1),
    to_connection("point_conv1", "BN3"),

    #CBAM
    to_ConvConvRelu(name='CBAM', s_filer=145, n_filer=(128, 128), offset="(2.0,0,0)", to="(BN3-east)", width=(8,8), height=8, depth=8, caption="CBAM"  ),
    to_connection("BN3", "CBAM"),

    to_SoftMax("soft1", 16, "(2.0,0,0)", "(CBAM-east)", caption="SOFT"  ),
    to_connection("CBAM", "soft1"),

    to_Sum("sum1", offset="(2.0,0,0)", to="(soft1-east)", radius=2.5, opacity=0.6),
    to_connection("soft1", "sum1"),
    to_end()
    ]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()