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
import tempfile
import shutil
from tqdm import tqdm

from image_processing.panel import generate_panel_blocks, generate_panel_blocks_by_ai

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

    # Create a temporary directory to store the processed panels
    with tempfile.TemporaryDirectory() as temp_panel_dir:
        print(f"Created temporary directory for panels: {temp_panel_dir}")

        for image_file in tqdm(input_files, desc="Processing Images"):
            try:
                # The image_file object from gr.Files has a .name attribute with the temp path
                original_filename = os.path.basename(image_file.name)
                filename_no_ext, file_ext = os.path.splitext(original_filename)

                # Read the image using OpenCV
                image = cv2.imread(image_file.name)
                if image is None:
                    print(f"Warning: Could not read image {original_filename}. Skipping.")
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

                if not panel_blocks:
                    print(f"Warning: No panels found in {original_filename}.")
                    continue

                # Determine the output path for the panels of this image
                if separate_folders:
                    # Create a sub-directory for each image
                    image_output_folder = os.path.join(temp_panel_dir, filename_no_ext)
                    os.makedirs(image_output_folder, exist_ok=True)
                else:
                    # Output all panels to the root of the temp directory
                    image_output_folder = temp_panel_dir

                # Save each panel block
                for i, panel in enumerate(panel_blocks):
                    if separate_folders:
                        # e.g., /tmp/xyz/image1/panel_0.png
                        panel_filename = f"panel_{i}{file_ext if file_ext else '.png'}"
                    else:
                        # e.g., /tmp/xyz/image1_panel_0.png
                        panel_filename = f"{filename_no_ext}_panel_{i}{file_ext if file_ext else '.png'}"

                    output_path = os.path.join(image_output_folder, panel_filename)
                    cv2.imwrite(output_path, panel)

            except Exception as e:
                print(f"Error processing {original_filename}: {e}")
                raise gr.Error(f"Failed to process {original_filename}: {e}")

        # After processing all images, check if any panels were generated
        if not os.listdir(temp_panel_dir):
            raise gr.Error("Processing complete, but no panels were extracted from any of the images.")
            
        # --- Create a zip file ---
        
        # Create a separate temporary directory to hold the final zip file.
        # Gradio will handle cleaning this up after serving the file to the user.
        zip_output_dir = tempfile.mkdtemp()
        
        # Define the base name for our archive (path + filename without extension)
        zip_path_base = os.path.join(zip_output_dir, "adenzu_output")
        
        # Create the zip file. shutil.make_archive will add the '.zip' extension.
        # The first argument is the full path for the output file (minus extension).
        # The third argument is the directory to be zipped.
        final_zip_path = shutil.make_archive(
            base_name=zip_path_base,
            format='zip',
            root_dir=temp_panel_dir
        )
        
        print(f"Created ZIP file at: {final_zip_path}")
        # The function returns the full path to the created zip file.
        # Gradio takes this path and provides it as a download link.
        return final_zip_path


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
                    split_joint = gr.Checkbox(label="Split Joint Panels", value=False)
                    fallback = gr.Checkbox(label="Fallback to Threshold Extraction", value=True)
                    output_mode = gr.Dropdown(
                        label="Output Mode",
                        choices=['bounding', 'masked'],
                        value='bounding'
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