#!/usr/bin/env python3

import argparse
import base64
import csv
import json
import os
import re
import sys
import time
import threading
import subprocess
import shlex
from io import BytesIO
from pathlib import Path
from typing import Optional, Dict, Literal
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

import requests
from PIL import Image

MAX_SLUG_LEN = 60

# Optional: load API key from ~/.ai_renamer.env (single line: OPENAI_API_KEY=...)
try:
    env_file = Path.home() / ".ai_renamer.env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k == "OPENAI_API_KEY" and v and not os.environ.get("OPENAI_API_KEY"):
                    os.environ["OPENAI_API_KEY"] = v
except Exception:
    pass

def image_to_base64(image_path):
    """Convert image file to base64 data URI"""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if needed (for RGBA images)
            if img.mode in ('RGBA', 'LA'):
                img = img.convert('RGB')
            
            # Resize if too large (optional, to reduce API costs)
            max_size = 1024
            if max(img.size) > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # Convert to base64
            buffer = BytesIO()
            img.save(buffer, format='JPEG', quality=85)
            img_data = buffer.getvalue()
            b64_string = base64.b64encode(img_data).decode('utf-8')
            
            return f"data:image/jpeg;base64,{b64_string}"
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return None

def call_openai_caption(image_b64_uri: str,
                        brand_hint: str,
                        keyword_hint: str,
                        business_hint: str,
                        location_hint: str,
                        mode: str = "service",
                        product_hint: str = "",
                        sku_hint: str = "",
                        category_hint: str = "",
                        emphasize_location: bool = True,
                        **_ignore) -> dict:
    """Call OpenAI API to generate image caption and filename"""
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    focus_geo = ("Include business type and location if relevant." if emphasize_location
                 else "Do not add city/state unless clearly present in the keyword.")
    
    mode_text = (
        "You are optimizing for a LOCAL SERVICE business (lead-gen). Filenames should bias toward service intent; alt text can include business type and city when appropriate."
        if mode == "service"
        else "You are optimizing for an E-COMMERCE PRODUCT page. Filenames should bias toward product/brand/model (and SKU when helpful). Avoid geo phrases unless present in the keyword."
    )

    prompt = f"""
{mode_text}
Analyze the image and return a concise JSON object optimized for image SEO.
Goals:
1) Identify the primary subject or product, including brand and model if visible.
2) Prefer the provided keyword as the filename head. If it doesn't match the image, choose the closest precise variant.
3) {focus_geo}

Return ONLY a valid JSON object with these fields:
- short_filename: 4-7 strong words, lowercase, hyphen-separated. No stopwords, no dates, no special characters. Keep under 60 chars after hyphenation.
- alt_text: 12-18 words, natural sentence that includes the brand or product.
- subject: main subject in the image
- product: product name if applicable
- brand: brand name if visible
- location: location if relevant
- action: action being performed if any

Hints:
- brand_hint: "{brand_hint}"
- keyword_hint: "{keyword_hint}"
- business_hint: "{business_hint}"
- location_hint: "{location_hint}"
- product_hint: "{product_hint}"
- sku_hint: "{sku_hint}"
- category_hint: "{category_hint}"

Respond with ONLY the JSON object, no other text.
"""

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_b64_uri
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300,
        "temperature": 0.1
    }

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code != 200:
            print(f"OpenAI API error: {response.status_code} - {response.text}")
            return {
                "short_filename": "error-processing-image",
                "alt_text": "Error processing image with AI",
                "subject": "unknown",
                "product": "",
                "brand": brand_hint or "",
                "location": location_hint or "",
                "action": ""
            }
        
        result = response.json()
        content = result["choices"][0]["message"]["content"].strip()
        
        # Try to parse JSON response
        try:
            data = json.loads(content)
            return data
        except json.JSONDecodeError:
            # If not valid JSON, try to extract it
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                    return data
                except json.JSONDecodeError:
                    pass
            
            # Fallback if JSON parsing fails
            print(f"Could not parse AI response as JSON: {content}")
            return {
                "short_filename": "ai-response-error",
                "alt_text": "AI response could not be parsed",
                "subject": "unknown",
                "product": "",
                "brand": brand_hint or "",
                "location": location_hint or "",
                "action": ""
            }
            
    except requests.exceptions.RequestException as e:
        print(f"Error calling OpenAI API: {e}")
        return {
            "short_filename": "api-error",
            "alt_text": "API error occurred",
            "subject": "unknown", 
            "product": "",
            "brand": brand_hint or "",
            "location": location_hint or "",
            "action": ""
        }

def process_single_image(image_file, args_dict):
    """Process a single image file"""
    print(f"Processing: {image_file.name}")
    
    try:
        # Convert image to base64
        b64 = image_to_base64(image_file)
        if not b64:
            print(f"  Error: Could not convert image to base64")
            return None
        
        # Call AI caption function
        data = call_openai_caption(
            b64,
            brand_hint=args_dict.get("brand", "") or args_dict.get("client", ""),
            keyword_hint=args_dict.get("keyword", ""),
            business_hint=args_dict.get("business", ""),
            location_hint=args_dict.get("location", ""),
            mode=args_dict.get("mode", "service"),
            product_hint=args_dict.get("product", ""),
            sku_hint=args_dict.get("sku", ""),
            category_hint=args_dict.get("category", ""),
            emphasize_location=bool(args_dict.get("emphasize_location", False)),
        )
        
        # Build filename core
        core = []
        kw = args_dict.get("keyword", "")
        if kw:
            core.extend(str(kw).split())
        
        mode = args_dict.get("mode", "service")
        if mode == "product":
            for hint in (args_dict.get("brand", ""), args_dict.get("product", ""), 
                       args_dict.get("sku", ""), args_dict.get("category", "")):
                if hint:
                    core.extend(str(hint).split())
        
        if data.get("short_filename"):
            core.extend(data["short_filename"].split("-"))
        else:
            for k in ("subject", "product", "brand", "action", "location"):
                if data.get(k):
                    core.extend(str(data[k]).split())
        
        if mode == "service":
            biz = args_dict.get("business", "")
            if biz:
                core.extend(str(biz).split())
            loc = args_dict.get("location", "")
            if bool(args_dict.get("emphasize_location", False)) and loc:
                core.extend(str(loc).split())
        
        # Clean and join core elements
        clean_core = [re.sub(r'[^a-z0-9]', '', word.lower()) for word in core if word]
        clean_core = [word for word in clean_core if len(word) > 1]  # Remove single chars
        new_filename = "-".join(clean_core[:8])  # Limit to 8 words
        
        # Ensure filename isn't empty
        if not new_filename:
            new_filename = f"image-{image_file.stem}"
        
        # Check if rename should actually happen
        if not args_dict.get("dry_run", True):
            # Actually rename file
            new_path = image_file.parent / f"{new_filename}{image_file.suffix}"
            counter = 1
            original_new_filename = new_filename
            
            # Handle filename conflicts
            while new_path.exists():
                new_filename = f"{original_new_filename}-{counter}"
                new_path = image_file.parent / f"{new_filename}{image_file.suffix}"
                counter += 1
            
            image_file.rename(new_path)
            print(f"  Renamed to: {new_path.name}")
        else:
            print(f"  Would rename to: {new_filename}{image_file.suffix}")
        
        # Return CSV data
        return {
            "original_filename": image_file.name,
            "new_filename": f"{new_filename}{image_file.suffix}",
            "alt_text": data.get("alt_text", ""),
            "subject": data.get("subject", ""),
            "product": data.get("product", ""),
            "brand": data.get("brand", ""),
            "location": data.get("location", ""),
            "action": data.get("action", ""),
            "mode": args_dict.get("mode", ""),
            "emphasize_location": str(bool(args_dict.get("emphasize_location", False))),
        }
        
    except Exception as e:
        print(f"  Error processing {image_file.name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_cli_script(args_dict):
    """Run the CLI logic with given arguments"""
    print(f"Processing images with arguments: {args_dict}")
    
    src_folder = Path(args_dict.get('src', ''))
    if not src_folder.exists():
        print(f"Error: Source folder {src_folder} does not exist")
        return 1
    
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Create ~/.ai_renamer.env with a line like: OPENAI_API_KEY=sk-...")
        return 1
    
    # Find image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    image_files = [f for f in src_folder.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"No image files found in {src_folder}")
        return 0
    
    print(f"Found {len(image_files)} image files in {src_folder}")
    
    csv_rows = []
    success_count = 0
    
    for image_file in image_files:
        try:
            result = process_single_image(image_file, args_dict)
            if result:
                csv_rows.append(result)
                success_count += 1
            else:
                print(f"  Failed to process {image_file.name}")
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            break
        except Exception as e:
            print(f"  Unexpected error with {image_file.name}: {e}")
    
    # Write CSV if requested
    if args_dict.get("csv", False) and csv_rows:
        csv_path = src_folder / "image_renaming_manifest.csv"
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                if csv_rows:
                    writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
                    writer.writeheader()
                    writer.writerows(csv_rows)
            print(f"CSV manifest written to: {csv_path}")
        except Exception as e:
            print(f"Error writing CSV: {e}")
    
    print(f"Processing complete! Successfully processed {success_count}/{len(image_files)} images")
    return 0

def parse_cli_args():
    """Parse command line arguments"""
    ap = argparse.ArgumentParser(description="AI Image Renamer CLI", argument_default="")
    ap.add_argument("--src", required=True, help="Source folder with images")
    ap.add_argument("--client", default="", help="Client or business name")
    ap.add_argument("--keyword", default="", help="Primary SEO keyword")
    ap.add_argument("--business", default="", help="Business type")
    ap.add_argument("--location", default="", help="Location info")
    ap.add_argument("--mode", default="service", choices=["service","product"], help="SEO mode: local service vs product page")
    ap.add_argument("--brand", default="", help="Brand name for product mode")
    ap.add_argument("--product", default="", help="Product/model descriptor for product mode")
    ap.add_argument("--sku", default="", help="SKU for product mode")
    ap.add_argument("--category", default="", help="Category for product mode, e.g. bump stops")
    ap.add_argument("--emphasize-location", action="store_true", help="Allow geo phrasing in SEO output (service mode)")
    ap.add_argument("--csv", action="store_true", help="Write CSV manifest")
    ap.add_argument("--dry-run", action="store_true", help="Preview only, don't rename files")
    ap.add_argument("--filename-from-alt", action="store_true", help="Build filename from alt text")
    
    args, _unknown = ap.parse_known_args()
    return vars(args)

# GUI Functions
def pick_folder(entry):
    path = filedialog.askdirectory()
    if path:
        entry.delete(0, tk.END)
        entry.insert(0, path)

def run_job(txt_log, btn_run, progress, values):
    btn_run.config(state="disabled")
    txt_log.delete("1.0", tk.END)
    progress.start(8)
    
    def worker():
        try:
            # Check API key
            if not os.environ.get("OPENAI_API_KEY"):
                messagebox.showerror(
                    "Missing API key",
                    "OPENAI_API_KEY is not set. Create ~/.ai_renamer.env with a line like\nOPENAI_API_KEY=sk-...\nThen click Run again."
                )
                progress.stop()
                btn_run.config(state="normal")
                return
            
            # Validate source folder
            if not values["src"] or not Path(values["src"]).exists():
                messagebox.showerror("Error", "Please select a valid source folder")
                progress.stop()
                btn_run.config(state="normal")
                return
            
            # Prepare arguments - only include non-empty values
            args_dict = {"src": values["src"]}
            
            # Add optional arguments only if they have values
            if values["client"]:
                args_dict["client"] = values["client"]
            if values["keyword"]:
                args_dict["keyword"] = values["keyword"]
            if values["business"]:
                args_dict["business"] = values["business"]
            if values["location"]:
                args_dict["location"] = values["location"]
            if values["mode"]:
                args_dict["mode"] = values["mode"]
            if values["brand"]:
                args_dict["brand"] = values["brand"]
            if values["product"]:
                args_dict["product"] = values["product"]
            if values["sku"]:
                args_dict["sku"] = values["sku"]
            if values["category"]:
                args_dict["category"] = values["category"]
            
            # Boolean arguments
            args_dict["emphasize_location"] = values["emphasize_location"]
            args_dict["csv"] = values["csv"]
            args_dict["dry_run"] = values["dry_run"]
            args_dict["filename_from_alt"] = values["filename_from_alt"]
            
            txt_log.insert(tk.END, f"Starting image processing...\n")
            txt_log.insert(tk.END, f"Arguments: {args_dict}\n\n")
            txt_log.see(tk.END)
            
            # Redirect stdout to capture print statements
            import io
            old_stdout = sys.stdout
            sys.stdout = captured_output = io.StringIO()
            
            try:
                # Run the CLI logic
                exit_code = run_cli_script(args_dict)
                
                # Get captured output
                output = captured_output.getvalue()
                sys.stdout = old_stdout
                
                # Display output in GUI
                txt_log.insert(tk.END, output)
                txt_log.see(tk.END)
                
                if exit_code == 0:
                    messagebox.showinfo("Done", "Completed without errors.")
                else:
                    messagebox.showerror("Error", f"Exited with code {exit_code}. Check the log.")
                    
            finally:
                sys.stdout = old_stdout
                
        except Exception as e:
            messagebox.showerror("Error", str(e))
            import traceback
            txt_log.insert(tk.END, f"\nError: {e}\n")
            txt_log.insert(tk.END, traceback.format_exc())
            txt_log.see(tk.END)
        finally:
            progress.stop()
            btn_run.config(state="normal")

    threading.Thread(target=worker, daemon=True).start()

def create_gui():
    root = tk.Tk()
    root.title("AI Image Renamer")

    frm = ttk.Frame(root, padding=12)
    frm.grid(row=0, column=0, sticky="nsew")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    # Inputs
    lbl_src = ttk.Label(frm, text="Source folder")
    ent_src = ttk.Entry(frm, width=60)
    btn_src = ttk.Button(frm, text="Browse", command=lambda: pick_folder(ent_src))

    lbl_client = ttk.Label(frm, text="Client")
    ent_client = ttk.Entry(frm, width=40)

    lbl_keyword = ttk.Label(frm, text="Primary keyword")
    ent_keyword = ttk.Entry(frm, width=40)

    lbl_business = ttk.Label(frm, text="Business type")
    ent_business = ttk.Entry(frm, width=40)

    lbl_location = ttk.Label(frm, text="Location")
    ent_location = ttk.Entry(frm, width=40)

    lbl_mode = ttk.Label(frm, text="SEO mode")
    mode_var = tk.StringVar(value="service")
    cmb_mode = ttk.Combobox(frm, textvariable=mode_var, values=["service","product"], width=12, state="readonly")

    lbl_brand = ttk.Label(frm, text="Brand (product)")
    ent_brand = ttk.Entry(frm, width=40)

    lbl_product = ttk.Label(frm, text="Product/model (product)")
    ent_product = ttk.Entry(frm, width=40)

    lbl_sku = ttk.Label(frm, text="SKU (product)")
    ent_sku = ttk.Entry(frm, width=40)

    lbl_cat = ttk.Label(frm, text="Category (product)")
    ent_cat = ttk.Entry(frm, width=40)

    var_csv = tk.BooleanVar(value=True)
    var_dry = tk.BooleanVar(value=True)
    var_from_alt = tk.BooleanVar(value=True)
    var_geo = tk.BooleanVar(value=True)
    chk_csv = ttk.Checkbutton(frm, text="Write CSV manifest", variable=var_csv)
    chk_dry = ttk.Checkbutton(frm, text="Dry run (preview only)", variable=var_dry)
    chk_from_alt = ttk.Checkbutton(frm, text="Filename from alt text", variable=var_from_alt)
    chk_geo = ttk.Checkbutton(frm, text="Emphasize location (service)", variable=var_geo)

    def refresh_mode_fields(*_):
        is_product = (mode_var.get() == "product")
        widgets_prod = [lbl_brand, ent_brand, lbl_product, ent_product, lbl_sku, ent_sku, lbl_cat, ent_cat]
        widgets_loc = [lbl_location, ent_location]
        if is_product:
            var_geo.set(False)
            for w in widgets_loc:
                w.grid_remove()
            for w in widgets_prod:
                w.grid()
            chk_geo.state(["disabled"])  # disable in product mode
        else:
            for w in widgets_prod:
                w.grid_remove()
            for w in widgets_loc:
                w.grid()
            chk_geo.state(["!disabled"])  # enable in service mode

    cmb_mode.bind("<<ComboboxSelected>>", refresh_mode_fields)

    btn_run = ttk.Button(
        frm,
        text="Run",
        command=lambda: run_job(
            txt_log, btn_run, progress, {
                "src": ent_src.get().strip(),
                "client": ent_client.get().strip(),
                "keyword": ent_keyword.get().strip(),
                "business": ent_business.get().strip(),
                "location": ent_location.get().strip(),
                "mode": mode_var.get(),
                "brand": ent_brand.get().strip(),
                "product": ent_product.get().strip(),
                "sku": ent_sku.get().strip(),
                "category": ent_cat.get().strip(),
                "emphasize_location": var_geo.get(),
                "csv": var_csv.get(),
                "dry_run": var_dry.get(),
                "filename_from_alt": var_from_alt.get(),
            }
        )
    )

    progress = ttk.Progressbar(frm, mode="indeterminate")

    txt_log = tk.Text(frm, height=18, width=90)
    scr = ttk.Scrollbar(frm, command=txt_log.yview)
    txt_log.configure(yscrollcommand=scr.set)

    # Layout
    lbl_src.grid(row=0, column=0, sticky="w")
    ent_src.grid(row=0, column=1, sticky="ew")
    btn_src.grid(row=0, column=2, padx=(6,0))

    lbl_client.grid(row=1, column=0, sticky="w", pady=(6,0))
    ent_client.grid(row=1, column=1, sticky="ew", pady=(6,0))

    lbl_keyword.grid(row=2, column=0, sticky="w")
    ent_keyword.grid(row=2, column=1, sticky="ew")

    lbl_business.grid(row=3, column=0, sticky="w")
    ent_business.grid(row=3, column=1, sticky="ew")

    lbl_location.grid(row=4, column=0, sticky="w")
    ent_location.grid(row=4, column=1, sticky="ew")

    lbl_mode.grid(row=5, column=0, sticky="w")
    cmb_mode.grid(row=5, column=1, sticky="w")

    lbl_brand.grid(row=6, column=0, sticky="w")
    ent_brand.grid(row=6, column=1, sticky="ew")

    lbl_product.grid(row=7, column=0, sticky="w")
    ent_product.grid(row=7, column=1, sticky="ew")

    lbl_sku.grid(row=8, column=0, sticky="w")
    ent_sku.grid(row=8, column=1, sticky="ew")

    lbl_cat.grid(row=9, column=0, sticky="w")
    ent_cat.grid(row=9, column=1, sticky="ew")

    chk_csv.grid(row=10, column=0, sticky="w", pady=(6,0))
    chk_dry.grid(row=10, column=1, sticky="w", pady=(6,0))
    chk_from_alt.grid(row=10, column=2, sticky="w", pady=(6,0))
    chk_geo.grid(row=11, column=0, sticky="w", pady=(0,6))

    btn_run.grid(row=12, column=0, sticky="w", pady=(8,0))
    progress.grid(row=12, column=1, sticky="ew", pady=(8,0))

    txt_log.grid(row=13, column=0, columnspan=2, sticky="nsew", pady=(10,0))
    scr.grid(row=13, column=2, sticky="ns", pady=(10,0))

    refresh_mode_fields()

    frm.columnconfigure(1, weight=1)
    frm.rowconfigure(13, weight=1)

    root.minsize(800, 420)
    return root

def main():
    """Main entry point - defaults to GUI mode"""
    
    # Check if user explicitly wants CLI mode
    if '--cli' in sys.argv:
        # Remove our custom flag before parsing
        sys.argv.remove('--cli')
        try:
            args_dict = parse_cli_args()
            exit_code = run_cli_script(args_dict)
            sys.exit(exit_code)
        except SystemExit as e:
            raise
        except Exception as e:
            print(f"Error in CLI mode: {e}")
            sys.exit(1)
    
    # Check if user provided CLI arguments without --gui flag
    elif any(arg.startswith('--') and arg not in ['--gui'] for arg in sys.argv[1:]):
        # User provided CLI arguments, try CLI mode
        try:
            # Remove --gui flag if present
            cli_args = [arg for arg in sys.argv if arg != '--gui']
            sys.argv = cli_args
            args_dict = parse_cli_args()
            exit_code = run_cli_script(args_dict)
            sys.exit(exit_code)
        except SystemExit as e:
            raise
        except Exception as e:
            print(f"Error in CLI mode: {e}")
            print("Falling back to GUI mode...")
            # Fall through to GUI mode
    
    # Default to GUI mode
    print("Starting GUI mode...")
    try:
        # Remove --gui flag if present
        if '--gui' in sys.argv:
            sys.argv.remove('--gui')
            
        root = create_gui()
        root.mainloop()
    except Exception as e:
        print(f"Error starting GUI: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to show error in a simple message box
        try:
            import tkinter.messagebox as mb
            mb.showerror("Startup Error", f"Failed to start GUI: {str(e)}")
        except:
            pass

if __name__ == "__main__":
    main()
