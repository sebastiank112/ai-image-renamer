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

def debug_print(message, debug_mode=True):
    """Print debug messages only if debug mode is enabled"""
    if debug_mode:
        print(message)

def image_to_base64(image_path, debug_mode=True):
    """Convert image file to base64 data URI"""
    try:
        debug_print(f"    DEBUG: Opening image file: {image_path}", debug_mode)
        
        with Image.open(image_path) as img:
            debug_print(f"    DEBUG: Image opened successfully, size: {img.size}, mode: {img.mode}", debug_mode)
            
            # Convert to RGB if needed (for RGBA images)
            if img.mode in ('RGBA', 'LA'):
                debug_print(f"    DEBUG: Converting {img.mode} to RGB", debug_mode)
                img = img.convert('RGB')
            
            # Resize if too large (optional, to reduce API costs)
            max_size = 1024
            if max(img.size) > max_size:
                debug_print(f"    DEBUG: Resizing from {img.size} to fit {max_size}px", debug_mode)
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # Convert to base64
            debug_print(f"    DEBUG: Converting to base64...", debug_mode)
            buffer = BytesIO()
            img.save(buffer, format='JPEG', quality=85)
            img_data = buffer.getvalue()
            b64_string = base64.b64encode(img_data).decode('utf-8')
            
            result = f"data:image/jpeg;base64,{b64_string}"
            debug_print(f"    DEBUG: Base64 conversion successful, length: {len(result)}", debug_mode)
            return result
            
    except Exception as e:
        print(f"    ERROR: Error converting image to base64: {e}")
        import traceback
        traceback.print_exc()
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
                        original_filename: str = "",
                        debug_mode: bool = True,
                        **_ignore) -> dict:
    """Call OpenAI API to generate image caption and filename"""
    
    debug_print(f"  DEBUG: Starting OpenAI API call...", debug_mode)
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(f"  ERROR: OPENAI_API_KEY environment variable not set")
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    debug_print(f"  DEBUG: API key found: {api_key[:8]}...{api_key[-4:] if len(api_key) > 12 else 'short'}", debug_mode)
    
    if not api_key.startswith("sk-"):
        print(f"  ERROR: API key doesn't start with 'sk-': {api_key[:20]}")
        raise ValueError("Invalid API key format")
    
    focus_geo = ("Include business type and location if relevant." if emphasize_location
                 else "Do not add city/state unless clearly present in the keyword.")
    
    mode_text = (
        "You are optimizing for a LOCAL SERVICE business (lead-gen). Filenames should bias toward service intent; alt text can include business type and city when appropriate."
        if mode == "service"
        else "You are optimizing for an E-COMMERCE PRODUCT page. Filenames should bias toward product/brand/model (and SKU when helpful). Avoid geo phrases unless present in the keyword."
    )

    prompt = f"""
{mode_text}
Analyze the image and return a concise JSON object optimized for LOCAL SEO.

Return ONLY a valid JSON object with these fields:
- filename: 4-6 words, lowercase, hyphen-separated, SEO-optimized filename
- alt_text: Natural sentence for image alt attribute (8-15 words)
- subject: main subject with specific details
- brand: equipment brand if visible
- action: specific activity or construction phase being shown

FILENAME REQUIREMENTS:
- Use hyphens between words
- Include specific activity + business type + location
- Keep it concise but descriptive
- Examples: "steel-beam-construction-rv-storage-loop-303", "concrete-foundation-boat-storage-surprise"

ALT_TEXT REQUIREMENTS:  
- Natural sentence for accessibility and SEO
- Include specific details about what's happening in the image
- Examples: "Steel beam construction at RV boat storage facility near Loop 303 and Grand Ave in Surprise Arizona"

LOCATION AND BUSINESS EXTRACTION:
From keyword_hint: "{keyword_hint}"
From business_hint: "{business_hint}" 
From location_hint: "{location_hint}"

Use these hints to create relevant filename and alt_text.

Context for THIS specific image:
- Original filename: {original_filename}

Hints:
- keyword_hint: "{keyword_hint}"
- business_hint: "{business_hint}"
- location_hint: "{location_hint}"
- brand_hint: "{brand_hint}"
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
        debug_print(f"  DEBUG: Making API request to OpenAI...", debug_mode)
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        debug_print(f"  DEBUG: API response status: {response.status_code}", debug_mode)
        
        if response.status_code != 200:
            print(f"  ERROR: OpenAI API error: {response.status_code}")
            print(f"  ERROR: Response text: {response.text}")
            return {
                "alt_text": f"Error processing image: API returned {response.status_code}",
                "subject": "unknown",
                "product": "",
                "brand": brand_hint or "",
                "action": ""
            }
        
        result = response.json()
        debug_print(f"  DEBUG: API response received, parsing...", debug_mode)
        
        content = result["choices"][0]["message"]["content"].strip()
        debug_print(f"  DEBUG: Raw AI response: {content[:200]}...", debug_mode)
        
        # Try to parse JSON response
        try:
            data = json.loads(content)
            debug_print(f"  SUCCESS: Parsed JSON response", debug_mode)
            debug_print(f"  Alt text: {data.get('alt_text', 'Missing')}", debug_mode)
            return data
        except json.JSONDecodeError:
            debug_print(f"  WARNING: Could not parse as JSON, trying to extract...", debug_mode)
            # If not valid JSON, try to extract it
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                    debug_print(f"  SUCCESS: Extracted and parsed JSON", debug_mode)
                    return data
                except json.JSONDecodeError:
                    pass
            
            # Fallback if JSON parsing fails
            print(f"  ERROR: Could not parse AI response as JSON")
            debug_print(f"  Full response: {content}", debug_mode)
            return {
                "alt_text": "AI response could not be parsed",
                "subject": "unknown",
                "product": "",
                "brand": brand_hint or "",
                "action": ""
            }
            
    except requests.exceptions.RequestException as e:
        print(f"  ERROR: Request exception: {e}")
        return {
            "alt_text": "API request failed",
            "subject": "unknown", 
            "product": "",
            "brand": brand_hint or "",
            "action": ""
        }
    except Exception as e:
        print(f"  ERROR: Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "alt_text": "Unexpected error occurred",
            "subject": "unknown",
            "product": "",
            "brand": brand_hint or "",
            "action": ""
        }

def process_single_image(image_file, args_dict):
    """Process a single image file"""
    debug_mode = args_dict.get("debug_mode", False)
    
    print(f"Processing: {image_file.name}")
    
    try:
        # Convert image to base64
        debug_print(f"  DEBUG: Starting image processing for {image_file.name}", debug_mode)
        debug_print(f"  DEBUG: Converting image to base64...", debug_mode)
        
        b64 = image_to_base64(image_file, debug_mode)
        if not b64:
            print(f"  ERROR: Could not convert image to base64")
            return None
        
        debug_print(f"  DEBUG: Successfully converted to base64 (length: {len(b64)})", debug_mode)
        
        # Call AI caption function
        debug_print(f"  DEBUG: About to call OpenAI API...", debug_mode)
        try:
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
                original_filename=image_file.name,
                debug_mode=debug_mode,
            )
            debug_print(f"  DEBUG: OpenAI call completed successfully", debug_mode)
        except Exception as api_error:
            print(f"  ERROR: OpenAI API call failed: {api_error}")
            if debug_mode:
                import traceback
                traceback.print_exc()
            # Create fallback data
            data = {
                "alt_text": f"API call failed: {str(api_error)[:50]}",
                "subject": "unknown",
                "product": "",
                "brand": args_dict.get("brand", ""),
                "action": ""
            }
        
        debug_print(f"  DEBUG: AI Response received", debug_mode)
        debug_print(f"  DEBUG: Filename: {data.get('filename', 'No filename')}", debug_mode)
        debug_print(f"  DEBUG: Alt text: {data.get('alt_text', 'No alt text')}", debug_mode)
        debug_print(f"  DEBUG: Subject: {data.get('subject', 'No subject')}", debug_mode)
        
        # Use AI-generated filename, with fallback
        ai_filename = data.get("filename", "")
        
        if ai_filename and ai_filename not in ["sample-image-filename", "error-processing-image", "ai-response-error", "api-error"]:
            # Use the AI-generated filename directly
            new_filename = ai_filename
            debug_print(f"  Using AI filename: {new_filename}", debug_mode)
        else:
            debug_print(f"  AI filename not available, building manually...", debug_mode)
            
            mode = args_dict.get("mode", "service")
            
            if mode == "product":
                # PRODUCT MODE: Focus on brand, product, vehicle/application
                core = []
                
                # Start with brand (most important for products)
                brand = args_dict.get("brand", "")
                if brand:
                    brand_parts = re.findall(r'\b\w+\b', brand.lower())
                    core.extend([part for part in brand_parts if len(part) > 1])
                
                # Add product name/type
                product = args_dict.get("product", "")
                if product:
                    product_parts = re.findall(r'\b\w+\b', product.lower())
                    # Filter out common product words but keep specific terms
                    product_stopwords = {'for', 'and', 'or', 'the', 'a', 'an', 'with'}
                    product_words = [part for part in product_parts if len(part) > 2 and part not in product_stopwords]
                    core.extend(product_words[:3])  # Limit to 3 most important terms
                
                # Add vehicle/application details from original filename
                original_stem = image_file.stem.lower()
                filename_parts = re.findall(r'\b\w+\b', original_stem)
                
                # Smart extraction of vehicle/application terms WITHOUT hardcoding
                vehicle_indicators = []
                for part in filename_parts:
                    # Extract meaningful terms based on intelligent patterns:
                    
                    # 1. Years - ANY 4-digit number that could reasonably be a vehicle year
                    if part.isdigit() and len(part) == 4:
                        year = int(part)
                        # From first automobiles (1880s) to reasonable future (next 10 years)
                        if 1880 <= year <= 2040:
                            vehicle_indicators.append(part)
                    
                    # 2. Generations/versions (ends with ordinal suffixes)
                    elif any(part.endswith(suffix) for suffix in ['th', 'nd', 'rd', 'st']) and len(part) <= 6:
                        vehicle_indicators.append(part)
                    
                    # 3. Model designations (intelligent pattern detection)
                    elif len(part) >= 2:
                        # Mixed alphanumeric (common in model names): f150, trx450, z71, etc.
                        has_letter = any(c.isalpha() for c in part)
                        has_number = any(c.isdigit() for c in part)
                        
                        if has_letter and has_number:
                            vehicle_indicators.append(part)
                        
                        # All caps acronyms/model codes (RZR, XJ, TJ, SX, etc.)
                        elif part.isupper() and 2 <= len(part) <= 6:
                            vehicle_indicators.append(part)
                        
                        # Common automotive suffixes/patterns
                        elif any(part.endswith(suffix) for suffix in ['r', 'x', 's', 'i', 'ti', 'si', 'gt', 'rs', 'rt']):
                            vehicle_indicators.append(part)
                    
                    # 4. Meaningful technical/specific terms
                    elif (len(part) > 3 and 
                          part not in [c.replace('-', '') for c in core] and
                          not any(stop in part for stop in ['and', 'or', 'the', 'for', 'with', 'from', 'that', 'this', 'will', 'when', 'where']) and
                          # Skip generic words but keep specific automotive/technical terms
                          not part in ['image', 'photo', 'picture', 'item', 'part', 'piece', 'product']):
                        vehicle_indicators.append(part)
                
                # Add SKU if meaningful
                sku = args_dict.get("sku", "")
                if sku and len(sku) > 2:
                    sku_clean = re.sub(r'[^a-z0-9]', '', sku.lower())
                    if sku_clean:
                        core.append(sku_clean)
                
                # Add vehicle indicators
                core.extend(vehicle_indicators[:2])
                
                # Add category for context
                category = args_dict.get("category", "")
                if category:
                    category_parts = re.findall(r'\b\w+\b', category.lower())
                    core.extend([part for part in category_parts if len(part) > 2][:1])
                
            else:
                # SERVICE MODE: Focus on activity, business type, location
                core = []
                
                # Start with keyword intelligently
                kw = args_dict.get("keyword", "")
                if kw:
                    keyword_parts = re.findall(r'\b\w+\b', kw.lower())
                    keyword_stopwords = {'and', 'or', 'the', 'a', 'an', 'in', 'at', 'on', 'near', 'by', 'for', 'with'}
                    keyword_words = [part for part in keyword_parts if len(part) > 2 and part not in keyword_stopwords]
                    core.extend(keyword_words[:2])
                
                # Add business type intelligently
                business = args_dict.get("business", "")
                if business:
                    business_parts = re.findall(r'\b\w+\b', business.lower())
                    business_stopwords = {'and', 'or', 'for', 'the', 'a', 'an', 'in', 'at', 'on', 'with', 'service', 'services', 'company', 'inc', 'llc'}
                    business_words = [part for part in business_parts if len(part) > 2 and part not in business_stopwords]
                    core.extend(business_words[:2])
                
                # Add location intelligently
                location = args_dict.get("location", "")
                if location:
                    location_parts = re.findall(r'\b\w+\b', location.lower())
                    location_stopwords = {'in', 'at', 'on', 'near', 'by', 'the', 'and', 'or', 'of', 'to', 'for', 'with'}
                    location_words = []
                    
                    for part in location_parts:
                        if len(part) > 2 and part not in location_stopwords:
                            if part.isdigit() and len(part) >= 2:
                                location_words.append(part)
                            elif any(road_type in part for road_type in ['st', 'rd', 'ave', 'blvd', 'way', 'dr', 'ln', 'loop', 'hwy']):
                                location_words.append(part)
                            elif not part.isdigit() and len(part) > 2:
                                location_words.append(part)
                    
                    core.extend(location_words[:3])
                
                # Add unique elements from original filename to differentiate
                original_stem = image_file.stem.lower()
                filename_parts = re.findall(r'\b\w+\b', original_stem)
                filename_stopwords = {'and', 'or', 'the', 'a', 'an', 'in', 'at', 'on', 'for', 'with', 'of', 'to', 'by', 'from', 'img', 'image', 'photo', 'pic', 'picture'}
                
                unique_elements = []
                for part in filename_parts:
                    if (len(part) > 3 and 
                        part not in filename_stopwords and 
                        not (part.isdigit() and len(part) < 3)):
                        
                        if part not in [c.replace('-', '') for c in core]:
                            unique_elements.append(part)
                
                core = unique_elements[:2] + core
            
            # Clean and create filename
            clean_core = [re.sub(r'[^a-z0-9-]', '', word) for word in core if word and len(word) > 1]
            # Remove duplicates while preserving order
            seen = set()
            unique_core = []
            for word in clean_core:
                if word not in seen:
                    unique_core.append(word)
                    seen.add(word)
            
            new_filename = "-".join(unique_core[:6])  # Limit to 6 words max
        
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
            "suggested_alt": data.get("alt_text", ""),  # For SEO reference
            "subject": data.get("subject", ""),
            "brand": data.get("brand", ""),
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
                    try:
                        messagebox.showinfo("Done", "Completed without errors.")
                    except Exception as msg_error:
                        print(f"Note: Processing completed successfully (GUI message error: {msg_error})")
                else:
                    try:
                        messagebox.showerror("Error", f"Exited with code {exit_code}. Check the log.")
                    except Exception as msg_error:
                        print(f"Error: Exited with code {exit_code}. Check the log. (GUI message error: {msg_error})")
                    
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

    var_csv = tk.BooleanVar(value=False)  # Unchecked by default
    var_dry = tk.BooleanVar(value=False)  # Unchecked by default
    var_geo = tk.BooleanVar(value=True)
    var_debug = tk.BooleanVar(value=False)  # Debug mode off by default
    
    chk_csv = ttk.Checkbutton(frm, text="CSV manifest", variable=var_csv)
    chk_dry = ttk.Checkbutton(frm, text="Dry run", variable=var_dry)
    chk_geo = ttk.Checkbutton(frm, text="Emphasize location", variable=var_geo)
    chk_debug = ttk.Checkbutton(frm, text="Debug output", variable=var_debug)

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
                "debug_mode": var_debug.get(),
            }
        )
    )

    progress = ttk.Progressbar(frm, mode="indeterminate")

    txt_log = tk.Text(frm, height=18, width=90)
    scr = ttk.Scrollbar(frm, command=txt_log.yview)
    txt_log.configure(yscrollcommand=scr.set)

    # Layout - Updated to fix checkbox positioning
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

    # Checkboxes - simple vertical stack (removed filename from alt)
    chk_csv.grid(row=10, column=0, sticky="w", pady=(6,2))
    chk_dry.grid(row=11, column=0, sticky="w", pady=2)
    chk_geo.grid(row=12, column=0, sticky="w", pady=2)
    chk_debug.grid(row=13, column=0, sticky="w", pady=(2,6))

    btn_run.grid(row=14, column=0, sticky="w", pady=(8,0))
    progress.grid(row=14, column=1, sticky="ew", pady=(8,0))

    txt_log.grid(row=15, column=0, columnspan=2, sticky="nsew", pady=(10,0))
    scr.grid(row=15, column=2, sticky="ns", pady=(10,0))

    refresh_mode_fields()

    frm.columnconfigure(1, weight=1)  # Main input column expands
    frm.rowconfigure(15, weight=1)    # Text log row expands

    root.minsize(800, 480)  # Slightly shorter since we removed a checkbox
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