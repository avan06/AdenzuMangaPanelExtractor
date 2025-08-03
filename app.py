"""
adenzu Eren Manga/Comics Panel Extractor (WebUI)
Copyright (C) 2025 avan

This program is a web interface for the adenzu library.
The core logic is based on adenzu Eren, the Manga Panel Extractor.
Copyright (c) 2023 Eren
"""
import gradio as gr
import os
import cv2
import numpy as np
import shutil
from tqdm import tqdm
from datetime import datetime

from image_processing.panel import generate_panel_blocks, generate_panel_blocks_by_ai
from manga_panel_processor import remove_border

# --- UI Description ---
DESCRIPTION = """
# adenzu Eren Manga/Comics Panel Extractor (WebUI)
Upload your manga or comic book images. This tool will automatically analyze and extract each panel.
You can choose between a Traditional algorithm or a AI-based model for processing.
Finally, all extracted panels are packaged into a single ZIP file for you to download.

The Core package author: **adenzu Eren** ([Original Project](https://github.com/adenzu/Manga-Panel-Extractor)).
"""

def process_images(
    input_files,
    method,
    separate_folders,
    rtl_order,
    remove_borders,
    # Traditional method params
    merge_mode,
    split_joint,
    fallback,
    output_mode,
    # AI method params (currently only merge_mode is used)
    # This structure allows for future AI-specific params
    progress=gr.Progress(track_tqdm=True)
):
    """
    Main processing function called by Gradio.
    It takes uploaded files and settings, processes them, and returns a zip file.
    """
    if not input_files:
        raise gr.Error("No images uploaded. Please upload at least one image.")

    # Create a unique, temporary sub-directory inside the 'output' folder for this run.
    main_output_dir = "output"
    os.makedirs(main_output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # All intermediate panel files will be stored here before being zipped.
    # This directory will be created inside 'output' and removed after zipping.
    panel_output_dir = os.path.join(main_output_dir, f"temp_panels_{timestamp}")
    os.makedirs(panel_output_dir)

    try:
        for image_file in tqdm(input_files, desc="Processing Images"):
            try:
                # The image_file object from gr.Files has a .name attribute with the temp path
                original_filename = os.path.basename(image_file.name)
                filename_no_ext, file_ext = os.path.splitext(original_filename)
                
                # Read image with Unicode path support to handle non-ASCII filenames.
                # Read the file into a numpy array first.
                stream = np.fromfile(image_file.name, np.uint8)
                # Decode the numpy array into an image.
                image = cv2.imdecode(stream, cv2.IMREAD_COLOR)

                if image is None:
                    print(f"Warning: Could not read or decode image {original_filename}. Skipping.")
                    continue

                # Select the processing function based on the chosen method
                if method == "Traditional":
                    panel_blocks = generate_panel_blocks(
                        image=image,
                        split_joint_panels=split_joint,
                        fallback=fallback,
                        mode=output_mode,
                        merge=merge_mode,
                        rtl_order=rtl_order
                    )
                elif method == "AI":
                    panel_blocks = generate_panel_blocks_by_ai(
                        image=image,
                        merge=merge_mode,
                        rtl_order=rtl_order
                    )
                else:
                    # Should not happen with Radio button selection
                    panel_blocks = []

                # If no panels were detected, use the original image as a single panel.
                if not panel_blocks:
                    print(f"Warning: No panels found in {original_filename}. Using the original image.")
                    panel_blocks = [image]

                # Determine the output path for the panels of this image
                if separate_folders:
                    # Create a sub-directory for each image
                    image_output_folder = os.path.join(panel_output_dir, filename_no_ext)
                    os.makedirs(image_output_folder, exist_ok=True)
                else:
                    # Output all panels to the root of the temp directory
                    image_output_folder = panel_output_dir

                # Save each panel block
                for i, panel in enumerate(panel_blocks):
                    if remove_borders:
                        panel = remove_border(panel)
                    
                    save_ext = file_ext if file_ext else '.png'
                    if separate_folders:
                        # e.g., /tmp/xyz/image_name/panel_0.png
                        panel_filename = f"panel_{i}{save_ext}"
                    else:
                        # e.g., /tmp/xyz/image_name_panel_0.png
                        panel_filename = f"{filename_no_ext}_panel_{i}{save_ext}"

                    output_path = os.path.join(image_output_folder, panel_filename)
                    
                    # Write image with Unicode path support.
                    # Encode the image to a memory buffer based on the file extension.
                    is_success, buffer = cv2.imencode(save_ext, panel)
                    if not is_success:
                        print(f"Warning: Could not encode panel {panel_filename}. Skipping.")
                        continue
                    # Write the buffer to a file using Python's standard I/O.
                    with open(output_path, 'wb') as f:
                        f.write(buffer)

            except Exception as e:
                print(f"Error processing {original_filename}: {e}")
                # Optionally, re-raise as a Gradio error to notify the user.
                # raise gr.Error(f"Failed to process {original_filename}: {e}")
                
        # After processing all images, check if any panels were generated
        if not os.listdir(panel_output_dir):
            raise gr.Error("Processing complete, but no panels were extracted from any of the images.")

        # --- Create a zip file in the 'output' directory ---
        zip_filename_base = f"adenzu_output_{timestamp}"

        # Define the full path for our archive (path + filename without extension).
        zip_path_base = os.path.join(main_output_dir, zip_filename_base)
        
        # Create the zip file from the temporary panel directory.
        final_zip_path = shutil.make_archive(
            base_name=zip_path_base,
            format='zip',
            root_dir=panel_output_dir
        )
        
        print(f"Created ZIP file at: {final_zip_path}")
        # The function returns the full path to the created zip file.
        # Gradio takes this path and provides it as a download link.
        return final_zip_path

    finally:
        # Clean up the temporary panel directory, leaving only the final ZIP file.
        # This block executes whether the 'try' block succeeds or fails.
        if os.path.exists(panel_output_dir):
            print(f"Cleaning up temporary panel directory: {panel_output_dir}")
            shutil.rmtree(panel_output_dir)


def main():
    """
    Defines and launches the Gradio interface.
    """
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            with gr.Column(scale=1):
                # --- Input Components ---
                input_files = gr.Files(
                    label="Upload Manga Pages",
                    file_types=["image"],
                    file_count="multiple"
                )

                method = gr.Radio(
                    label="Processing Method",
                    choices=["Traditional", "AI"],
                    value="Traditional",
                    interactive=True
                )

                # --- Output Options ---
                gr.Markdown("### Output Options")
                separate_folders = gr.Checkbox(
                    label="Create a separate folder for each image inside the ZIP",
                    value=True,
                    info="If unchecked, all panels will be in the root of the ZIP, with filenames prefixed by the original image name."
                )

                rtl_order = gr.Checkbox(
                    label="Right-to-Left (RTL) Reading Order",
                    value=True, # Default to True for manga
                    info="Check this for manga that is read from right to left. Uncheck for western comics."
                )

                remove_borders = gr.Checkbox(
                    label="Attempt to remove panel borders",
                    value=False,
                    info="Crops the image to the content area. May not be perfect for all images."
                )

                # --- Shared Parameters ---
                gr.Markdown("### Shared Parameters")
                merge_mode = gr.Dropdown(
                    label="Merge Mode",
                    choices=['none', 'vertical', 'horizontal'],
                    value='none',
                    info="How to merge detected panels before saving."
                )

                # --- Method-specific Parameters ---
                with gr.Group(visible=True) as traditional_params:
                    gr.Markdown("### Traditional Method Parameters")
                    split_joint = gr.Checkbox(
                        label="Split Joint Panels",
                        value=False,
                        info="For panels that are touching or share a single border line. This algorithm actively tries to draw a separation line between them. Useful if multiple panels are being detected as one large block, but may occasionally split a single large panel by mistake."
                    )
                    fallback = gr.Checkbox(
                        label="Fallback to Threshold Extraction",
                        value=True,
                        info="If the main algorithm fails to find multiple panels (e.g., on a borderless page or a full-bleed splash page), this enables a secondary, simpler extraction method. It's a 'safety net' that can find panels when the primary method cannot."
                    )
                    output_mode = gr.Dropdown(
                        label="Output Mode",
                        choices=['bounding', 'masked'],
                        value='bounding',
                        info="bounding: Crops a rectangular area around each panel. Best for general use. \nmasked: Crops along the exact, non-rectangular shape of the panel, filling the outside with a background color. Best for irregularly shaped panels."
                    )

                with gr.Group(visible=False) as ai_params:
                    gr.Markdown("### AI Method Parameters")
                    gr.Markdown("_(Currently, only the shared 'Merge Mode' parameter is used by the AI method.)_")

                # --- UI Logic to show/hide parameter groups ---
                def toggle_parameter_visibility(selected_method):
                    if selected_method == "Traditional":
                        return gr.update(visible=True), gr.update(visible=False)
                    elif selected_method == "AI":
                        return gr.update(visible=False), gr.update(visible=True)
                
                method.change(
                    fn=toggle_parameter_visibility,
                    inputs=method,
                    outputs=[traditional_params, ai_params]
                )

                # --- Action Button ---
                generate_button = gr.Button("Generate Panels", variant="primary")

            with gr.Column(scale=1):
                # --- Output Component ---
                output_zip = gr.File(label="Download ZIP")

        # --- Button Click Action ---
        generate_button.click(
            fn=process_images,
            inputs=[
                input_files,
                method,
                separate_folders,
                rtl_order,
                remove_borders,
                merge_mode,
                split_joint,
                fallback,
                output_mode
            ],
            outputs=output_zip
        )

    demo.launch(inbrowser=True)

if __name__ == "__main__":
    main()