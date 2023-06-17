import os
import torch
import math
import numpy as np
import gradio as gr
import time 

import modules.scripts as scripts
import modules.sd_samplers_common as sd
import modules.sd_samplers_kdiffusion as kd
import modules.shared as shared
from modules.script_callbacks import CFGDenoiserParams, on_cfg_denoiser, CFGDenoisedParams, on_cfg_denoised, remove_current_script_callbacks
from modules.shared import sd_model, device, state, opts

from PIL import Image
from modules.images import save_image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Script(scripts.Script):

    def title(self):
        return "Latentshop"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("Latentshop", open=False, elem_id="#Latentshop"):

            with gr.Tab("Schedule"):
                with gr.Box():
                    with gr.Row(variant='compact', elem_id="#LatentshopTBar"):
                        z_enabled = gr.Checkbox(label='Enable', value=False, min_width=25)                                                    
                        z_reverse = gr.Checkbox(label='Reverse', value=False, min_width=25) 
                        z_clamp_start = gr.Checkbox(label='Clamp Start', value=True, visible=True, min_width=25)                          
                        z_clamp_end = gr.Checkbox(label='Clamp End', value=True, visible=True, min_width=25)                          
                        z_when = gr.Dropdown(["before", "during", "after"], value="after", label="Callback", show_label=True)                                                                                   

                    with gr.Row().style(equal_height=True):                 
                        with gr.Column(scale=3, min_width=220):                                               
                            z_amount = gr.Slider(minimum=-1.0, maximum=1.0, step=.001, value=0.025, label="Total Amount")
                            z_exponent = gr.Slider(minimum=0.0, maximum=8.0, step=.01, value=2.00, label="Exponent")
                            with gr.Row():
                                z_start = gr.Slider(minimum=0.0, maximum=1.0, step=.01, value=0.0, label="Start")
                                z_end = gr.Slider(minimum=0.0, maximum=1.0, step=.01, value=1.0, label="End")
                            z_offset = gr.Slider(minimum=-1.0, maximum=1.0, step=.001, value=0.00, label="Offset")
                                                    
                        with gr.Column(scale=1, min_width=275):  
                            preview = self.visualize(False, 'after', False, 0, 2.0, 0, False, False, 0, 1)                                 
                            z_vis = gr.Plot(value=preview, elem_id='LatentshopVis', show_label=False)                      
                                            
            with gr.Tab("Adjustment"):
                with gr.Box():
                    with gr.Row():
                        
                        z_mode = gr.Dropdown(["contrast","fade"], value="contrast", label="Contrast Mode", show_label=True) 

                    with gr.Row(elem_id='LatentshopAdj'):
                        with gr.Column(scale=8, min_width=100):   
                            z_brightness = gr.Slider(minimum=-1.0, maximum=1.0, step=.05, value=0.0, label="Brightness")                            
                            z_r = gr.Slider(minimum=-2, maximum=2, step=.05, value=0.0, label="R")
                            z_g = gr.Slider(minimum=-2, maximum=2, step=.05, value=0.0, label="G")
                            z_b = gr.Slider(minimum=-2, maximum=2, step=.05, value=0.0, label="B")
                        with gr.Column(scale=1, min_width=25):  
                            z_swatch = gr.ColorPicker(value="#808080", label="Grey Latent", interactive=False, elem_id="z_color_swatch")
                        with gr.Column(scale=8, min_width=100):                                                      
                            z_ch0 = gr.Slider(minimum=-1.0, maximum=1.0, step=.05, value=1.00, label="Channel 0")
                            z_ch1 = gr.Slider(minimum=-1.0, maximum=1.0, step=.05, value=1.00, label="Channel 1")
                            z_ch2 = gr.Slider(minimum=-1.0, maximum=1.0, step=.05, value=1.00, label="Channel 2")
                            z_ch3 = gr.Slider(minimum=-1.0, maximum=1.0, step=.05, value=1.00, label="Channel 3")

                    z_skip = gr.Slider(minimum=0.0, maximum=1.0, step=.01, value=0.0, label="Skip End", visible=False)      

            with gr.Tab("Presets"):
                z_preset_name = gr.Dropdown([], label="Preset", value="DeCo", visible=False)
                with gr.Box():          
                    gr.Examples(
                        [
                        ["BlackSeed",  "fade",       "after",  True,  True, True, .75,  1,   0, 0.10, .5,    -1,  0,  0,     0,    1,  1,  1,  1],
                        ["WhiteSeed",  "fade",       "after",  True,  True, True, .75,  1,   0, 0.10, .5,     1,  0,  0,     0,    1,  1,  1,  1],                        
                        ["Darken",     "contrast",   "after",  False, True, True, .15,  2,   0, 0.20, .9,    -1,  0,  0,     0,    1, .5, .5, .5],                                        
                        ["Brighten",   "contrast",   "after",  False, True, True, .15,  2,   0, 0.20, .9,     1,  0,  0,     0,    1, .5, .5, .5],
                        ["Flash",      "fade",       "after",  True,  True, True, .66,  1,   0,    1, .6,     1,  0,  0,     0,    1,  1,  1,  1],
                        ["deTint",     "contrast",   "before", True,  True, True, .1,   0,   0,    1, .9,     0, -2, -1.35,  0,   -1,  1,  1,  1],
                        ["deBurr",     "fade",       "after",  False, True, True, .07, .55,  0,   .3, .93,   .1,  0,  0,     0,    0,  1,  1,  1],
                        ["deContrast", "fade",       "after",  False, True, True, .1,   4,   0, 0.50, .9,     1,  0,  0,     0,    1,  1,  1,  1],                        
                        ],
                        [z_preset_name, z_mode, z_when, z_reverse, z_clamp_start, z_clamp_end, z_amount, z_exponent, z_offset, z_start, z_end, z_brightness, z_r, z_g, z_b, z_ch0, z_ch1, z_ch2, z_ch3],
                        label="Presets", elem_id="PresetsTable"
                        )

        vis_args = [z_enabled, z_when, z_reverse, z_amount, z_exponent, z_offset, z_clamp_start, z_clamp_end, z_start, z_end]
        for vis_arg in vis_args:
            vis_arg.change(fn=self.visualize, show_progress=False, inputs=vis_args, outputs=[z_vis])
            if isinstance(vis_arg, gr.components.Slider) : vis_arg.release(fn=self.visualize, show_progress=False, inputs=vis_args, outputs=[z_vis])
            # Not using only release, cuz in current version it doesn't fire when slider is updated through textbox/arrows
            # But still using release, cuz .change does not fire when mouse pointer is outside slider width (so it doesn't update when you pull past 0 or 1 and then release)

        vis_col_args = [z_brightness, z_r, z_g, z_b]
        for vis_col_arg in vis_col_args:
            vis_col_arg.change(fn=self.visualize_col, show_progress=False, inputs=vis_col_args, outputs=[z_swatch])
            vis_col_arg.release(fn=self.visualize_col, show_progress=False, inputs=vis_col_args, outputs=[z_swatch])

        self.infotext_fields = []        
        self.infotext_fields.extend([
            (z_enabled, "Latentshop"),
            (z_when, "When"),
            (z_reverse, "Reverse"),
            (z_clamp_start, "Clamp_Start"),
            (z_clamp_end, "Clamp_End"),
            (z_mode, "Mode"),
            (z_amount, "Amount"),
            (z_exponent, "Exponent"),
            (z_offset, "Offset"),
            (z_start, "Start"),
            (z_end, "End"),
            (z_brightness, "Brightness"),
            (z_r, "R"),
            (z_g, "G"),
            (z_b, "B"),
            (z_ch0, "Ch0"),
            (z_ch1, "Ch1"),
            (z_ch2, "Ch2"),
            (z_ch3, "Ch3"),
            (z_skip, "Skip")
        ])
        self.paste_field_names = []
        for _, field_name in self.infotext_fields:
            self.paste_field_names.append(field_name)

        return [z_enabled, z_mode, z_when, z_amount, z_reverse, z_exponent, z_offset, z_start, z_end, z_brightness, z_clamp_start, z_clamp_end, z_ch0, z_ch1, z_ch2, z_ch3, z_r, z_g, z_b, z_skip]
    

    def process(self, p, z_enabled, z_mode, z_when, z_amount, z_reverse, z_exponent, z_offset, z_start, z_end, z_brightness, z_clamp_start, z_clamp_end, z_ch0, z_ch1, z_ch2, z_ch3, z_r, z_g, z_b, z_skip):
        
        stop_at = z_skip
        self.stop_at = z_skip
        grey_latent = None
        self.grey_latent = None
        schedule = None
        self.schedule = None
        channel_mix = None
        self.channel_mix = None
        contrast_mode = None
        self.contrast_mode = None

        self.p = p

        z_amount = getattr(p, 'LS_amount', z_amount)
        z_exponent = getattr(p, 'LS_exponent', z_exponent)
        z_start = getattr(p, 'LS_start', z_start)
        z_end = getattr(p, 'LS_end', z_end)
        z_offset = getattr(p, 'LS_offset', z_offset)
        z_mode = getattr(p, 'LS_mode', z_mode)

        
        def original_callback_state(self, d):
            step = d['i']
            latent = d["denoised"]
            if opts.live_preview_content == "Combined":
                sd.store_latent(latent)
            self.last_latent = latent        
            if self.stop_at is not None and step > self.stop_at:
                raise sd.InterruptedException
            state.sampling_step = step
            shared.total_tqdm.update()

        def new_callback_state(self, d):
            step = d['i']   
            x = d['x']
            ch = channel_mix
            t = schedule[step]

            if contrast_mode == "fade" : 
                x0 = torch.lerp(x[:, 0:1, :, :], grey_latent[:, 0:1, :, :], t * ch[0])
                x1 = torch.lerp(x[:, 1:2, :, :], grey_latent[:, 1:2, :, :], t * ch[1])
                x2 = torch.lerp(x[:, 2:3, :, :], grey_latent[:, 2:3, :, :], t * ch[2])
                x3 = torch.lerp(x[:, 3:4, :, :], grey_latent[:, 3:4, :, :], t * ch[3])
                x = torch.cat([x0, x1, x2, x3], dim=1, out=x)
            else:
                for i in range(4):
                    x[:, i, :, :] -= grey_latent[:, i, :, :]
                    signs = torch.sign(x[:, i, :, :])
                    mx = torch.max(torch.abs(x[:, i, :, :]))
                    x[:, i, :, :] /= mx
                    tmp = torch.pow(torch.abs(x[:, i, :, :]), 1 + t * ch[i], out=x[:, i, :, :])
                    x[:, i, :, :] *= mx
                    x[:, i, :, :] *= signs
                    x[:, i, :, :] += grey_latent[:, i, :, :]

            if stop_at is not None and step > (1 - stop_at) * (state.sampling_steps - 1):
                print("Aborted by Latentshop")
                self.last_latent = state.current_latent
                raise sd.InterruptedException

            return original_callback_state(self, d)            

        if hasattr(self, 'callbacks_added'):
            remove_current_script_callbacks()
            kd.KDiffusionSampler.callback_state = original_callback_state   
            print('callbacks_removed')  

        if z_enabled:
   
            grey_latent = self.make_grey_latent(p, z_brightness, z_r, z_g, z_b)
            self.grey_latent = grey_latent   
            schedule = self.make_schedule(p.steps, z_when, z_reverse, z_amount, z_exponent, z_offset, z_clamp_start, z_clamp_end, z_start, z_end )
            self.schedule = schedule
            channel_mix = [z_ch0, z_ch1, z_ch2, z_ch3]
            self.channel_mix = channel_mix
            contrast_mode = z_mode
            self.contrast_mode = contrast_mode
        
            if z_when == "before" : 
                on_cfg_denoiser(self.denoise_callback)
            elif z_when == "after" : 
                on_cfg_denoised(self.denoise_callback)
            else :
                kd.KDiffusionSampler.callback_state = new_callback_state
            print('callbacks_added')
            self.callbacks_added = True    

            p.extra_generation_params.update({
                "Latentshop": z_enabled,
                "When": z_when,
                "Reverse": z_reverse,
                "Clamp_Start": z_clamp_start,
                "Clamp_End": z_clamp_end,
                "Mode": z_mode,
                "Amount": z_amount,
                "Exponent": z_exponent,
                "Offset": z_offset,
                "Start": z_start,
                "End": z_end,
                "Brightness": z_brightness,
                "R": z_r,
                "G": z_g,
                "B": z_b,
                "Ch0": z_ch0,
                "Ch1": z_ch1,
                "Ch2": z_ch2,
                "Ch3": z_ch3,
                "Skip": z_skip
            })
        return

    def denoise_callback(self, params):
        step = params.sampling_step
        x = params.x        
        ch = self.channel_mix
        t = self.schedule[step]

        if self.contrast_mode == "fade" : 
            x0 = torch.lerp(x[:, 0:1, :, :], self.grey_latent[:, 0:1, :, :], t * ch[0])
            x1 = torch.lerp(x[:, 1:2, :, :], self.grey_latent[:, 1:2, :, :], t * ch[1])
            x2 = torch.lerp(x[:, 2:3, :, :], self.grey_latent[:, 2:3, :, :], t * ch[2])
            x3 = torch.lerp(x[:, 3:4, :, :], self.grey_latent[:, 3:4, :, :], t * ch[3])
            x = torch.cat([x0, x1, x2, x3], dim=1, out=x)
        else:
            for i in range(4):
                x[:, i, :, :] -= self.grey_latent[:, i, :, :]
                signs = torch.sign(x[:, i, :, :])
                mx = torch.max(torch.abs(x[:, i, :, :]))
                x[:, i, :, :] /= mx
                tmp = torch.pow(torch.abs(x[:, i, :, :]), 1 + t * ch[i], out=x[:, i, :, :])
                x[:, i, :, :] *= mx
                x[:, i, :, :] *= signs
                x[:, i, :, :] += self.grey_latent[:, i, :, :]

    def make_grey_latent(self, p, brightness, red, green, blue):
        guide = torch.zeros(1,3,p.height,p.width,device=device)
        if not shared.cmd_opts.no_half:
            guide = guide.half()
        guide[:,0,:,:] += red
        guide[:,1,:,:] += green
        guide[:,2,:,:] += blue
        guide += brightness
        guide = p.sd_model.get_first_stage_encoding(p.sd_model.encode_first_stage(guide))
        midx = int(guide.shape[2]/2)
        midy = int(guide.shape[3]/2)
        grey = torch.ones_like(guide)
        grey[:,0,:,:] *= guide[:,0,midx,midy]
        grey[:,1,:,:] *= guide[:,1,midx,midy]
        grey[:,2,:,:] *= guide[:,2,midx,midy]
        grey[:,3,:,:] *= guide[:,3,midx,midy]  
        return grey

    def make_schedule(self, steps, when, reverse, amount, exponent, offset, clamp_start, clamp_end, start, end ):
        values = []
        start = int(start * (steps-1))
        end = int(end * (steps-1))
        if start >= end: start = end - 1
        for i in range(steps):
            t = (i - start) / (end - start)
            # uncomment for cosine, but let's face it, this is only cosmetic:
            # t = 0.5 * (1 - math.cos(t * math.pi))
            if reverse: 
                t = 1 - t
            t **= exponent    
            t *= amount

            t = 0 if not clamp_start and i < start else t
            t = 0 if not clamp_end and i > end else t

            t += offset

            t = 0 if clamp_start and i < start else t
            t = 0 if clamp_end and i > end else t

            values.append(t)
        return values

    def visualize(self, enabled, when, reverse, amount, exponent, offset, clamp_start, clamp_end, start, end):
        try:
            steps = 51
            values = self.make_schedule(steps, when, reverse, amount, exponent, offset, clamp_start, clamp_end, start, end )
            mean = sum(values)/steps
            plot_color = (0.5, 0.5, 0.5, 0.5) if not enabled else (0.3, 0.8, 0.0, 0.75) if mean >= 0 else (1.0, 0.3, 0.0, 0.75) 
            plt.rcParams.update({
                "text.color":  plot_color, 
                "axes.labelcolor":  plot_color, 
                "axes.edgecolor":  plot_color, 
                "figure.facecolor":  (0.0, 0.0, 0.0, 0.0),  
                "axes.facecolor":    (0.0, 0.0, 0.0, 0.0),  
                "ytick.labelsize": 6,
                "ytick.labelcolor": plot_color,
                "ytick.color": plot_color,
            })
            
            fig, ax = plt.subplots(figsize=(2.15, 2.00),layout="constrained")

            ax.plot(range(steps), values, color=plot_color)

            ax.axhline(y=0, color=plot_color, linestyle='dotted')
            ax.set_xlabel('Adjustment Schedule')
            ax.tick_params(right=True)
            ax.set_xticks([])
            ax.set_ylim([-.5,.5])
            ax.set_xlim([0,steps-1])

            # plt.tight_layout()
            plt.close()
            return fig   
        except:
            return   

    def visualize_col(self, brightness, red, green, blue):
        r = max(0, min(int(255 * (red + brightness + 1) / 2), 255))
        g = max(0, min(int(255 * (green + brightness + 1) / 2), 255))
        b = max(0, min(int(255 * (blue + brightness + 1) / 2), 255))
        return "#{:02x}{:02x}{:02x}".format(r, g, b)

def xyz():
    for scriptDataTuple in scripts.scripts_data:
        if os.path.basename(scriptDataTuple.path) == 'xyz_grid.py':
            xy_grid = scriptDataTuple.module

            def confirm_mode(p, xs):
                for x in xs:
                    if x not in ['contrast','fade']:
                        raise RuntimeError(f'Invalid op: {x}')

            z_amount = xy_grid.AxisOption(
                '[Latentshop] Amount',
                float,
                xy_grid.apply_field('LS_amount')
            )
            z_exponent = xy_grid.AxisOption(
                '[Latentshop] Schedule Exponent',
                float,
                xy_grid.apply_field('LS_exponent')
            )            
            z_start = xy_grid.AxisOption(
                '[Latentshop] Schedule Start',
                float,
                xy_grid.apply_field('LS_start')
            )
            z_end = xy_grid.AxisOption(
                '[Latentshop] Schedule End',
                float,
                xy_grid.apply_field('LS_end')
            )           
            z_offset = xy_grid.AxisOption(
                '[Latentshop] Schedule Offset',
                float,
                xy_grid.apply_field('LS_offset')
            )                    
            z_mode = xy_grid.AxisOption(
                '[Latentshop] Contrast Mode',
                str,
                xy_grid.apply_field('LS_mode'),
                confirm=confirm_mode
            )

            xy_grid.axis_options.extend([
                z_amount,
                z_exponent,
                z_start,
                z_end,
                z_offset,
                z_mode,
            ])
try:
    xyz()
except Exception as e:
    print(f'Error adding XYZ plot options for Latentshop, this should break everything!', e)
