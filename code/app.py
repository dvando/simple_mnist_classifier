import torch
import kornia as K
import gradio as gr
import pandas as pd
import cv2
import torch.nn.functional as F

from models import MyNet

class Predictor():
    def __init__(self) -> None:
        self.model = MyNet().cuda()
        ckpt= torch.load('../model/myNet.pth', map_location='cpu')
        self.model.load_state_dict(ckpt)        

    def inference(self, img):
        img = cv2.resize(img, (28,28))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = K.utils.image_to_tensor(img).float().unsqueeze(0).cuda()
        with torch.no_grad():
            self.model.eval()
            feat1, feat2, feat3, feat4, out = self.model(img)
        
        feats = [feat1, feat2, feat3, feat4]
        featts = []
        out = out.squeeze().argmax()
        for i, f in enumerate(feats):
            f =  F.interpolate(f, (28,28))
            feats[i] = (f.squeeze()[i]*255).cpu().to(torch.uint8).numpy()
            if i == 1:
                [featts.append(c) for c in (f.squeeze()*255).cpu().to(torch.uint8).numpy()]
        
        for i, (s, t) in enumerate(zip(feats, featts)):
            feats[i] = cv2.cvtColor(s, cv2.COLOR_GRAY2BGR) 
            featts[i] = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR) 

        return feats[0], feats[1], feats[2], feats[3], featts[0], featts[1], featts[2], featts[3], out.item()


def main():
    predictor = Predictor()
    demo = gr.Blocks()

    with demo:
        gr.Markdown("""
        # Simple MNIST Digit Classification by Agung Aldevando
        """)
        with gr.Row():
            img_input= gr.Image()
            output = gr.Textbox(label= 'Prediction Result')

        gr.Markdown('Several channel in different layer')
        with gr.Row():
            feat1 = gr.Image()
            feat2 = gr.Image()
            feat3 = gr.Image()
            feat4 = gr.Image()

        gr.Markdown('Several channel in the same layer')
        with gr.Row():
            feat2_1 = gr.Image()
            feat2_2 = gr.Image()
            feat2_3 = gr.Image()
            feat2_4 = gr.Image()

        bt_inference = gr.Button('Predict')
        bt_inference.click(predictor.inference, 
                           inputs=img_input, 
                           outputs=[feat1, feat2, feat3, feat4, feat2_1, 
                                    feat2_2, feat2_3, feat2_4, output])

    demo.launch(share=True)

if __name__ == '__main__':
    main()
