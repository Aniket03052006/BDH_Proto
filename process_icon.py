from PIL import Image
import os

def process_icon(input_path, output_path):
    try:
        img = Image.open(input_path)
        img = img.convert("RGBA")
        datas = img.getdata()
        
        # Determine background color from corners
        corners = [
            datas[0], 
            datas[img.width-1], 
            datas[(img.height-1)*img.width], 
            datas[(img.height-1)*img.width + img.width-1]
        ]
        
        # Most frequent corner color
        bg_color = max(set(corners), key=corners.count)
        print(f"Detected background color: {bg_color}")
        
        new_data = []
        tolerance = 30
        
        for item in datas:
            # Check if color is close to background color
            if (abs(item[0] - bg_color[0]) < tolerance and 
                abs(item[1] - bg_color[1]) < tolerance and 
                abs(item[2] - bg_color[2]) < tolerance):
                new_data.append((255, 255, 255, 0)) # Transparent
            else:
                new_data.append(item)
                
        img.putdata(new_data)
        
        # Resize to standard icon sizes if needed, but for now just save original size
        # Maybe create a square version if not square?
        # Let's just save as is for now.
        
        img.save(output_path, "PNG")
        print(f"Successfully saved transparent icon to {output_path}")
        
    except Exception as e:
        print(f"Error processing icon: {e}")

if __name__ == "__main__":
    if os.path.exists("Icon.jpeg"):
        process_icon("Icon.jpeg", "icon.png")
    else:
        print("Icon.jpeg not found!")
