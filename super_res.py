import argparse
import torch
import numpy as np
from math import log10

from PIL import Image
from torchvision.transforms import ToTensor


def main():
    to_upscale = Image.open(args.input).convert('YCbCr')
    y, cb, cr = to_upscale.split()

    model_in = torch.load(args.model)
    image_to_tensor = ToTensor()
    input = image_to_tensor(y).view(1, -1, y.size[1], y.size[0])

    if args.cuda and torch.cuda.is_available():
        model_in = model_in.cuda()
        input = input.cuda()

    out = model_in(input)
    out = out.cpu()

    out_img_y = out[0].detach().numpy()
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    y_input = out_img_y
    out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

    out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
    out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')

    if args.compare is not "None":
        to_compare = Image.open(args.compare).convert('YCbCr')
        y_comp, _, _ = to_compare.split()
        y_compare = np.array(y_comp)

        mse = np.mean((y_compare - y_input) ** 2)
        if mse == 0:
            psnr = 100
        else:
            PIXEL_MAX = 255.0
            psnr = 20 * log10(PIXEL_MAX / np.sqrt(mse))
        print("PSNR of output wrt compared is: " + str(psnr) + "\n")

    out_img.save(args.output)
    print('output image saved to ', args.output)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DRRA Project")
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--compare', type=str, default="None")
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--cuda', action='store_true', help="enable cuda?")
    args = parser.parse_args()

    print("Input args:" + str(args))

    main()