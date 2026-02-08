from PIL import Image, ImageDraw

def create_icon():
    # Create a 256x256 image with a dark background
    size = (256, 256)
    image = Image.new('RGBA', size, (15, 23, 42, 255)) # Slate-900
    draw = ImageDraw.Draw(image)
    
    # Draw a stylized "Q" or microphone
    # Circular base
    draw.ellipse([20, 20, 236, 236], outline=(59, 130, 246, 255), width=8) # Blue-500
    
    # Soundwave bars
    colors = [(59, 130, 246), (96, 165, 250), (249, 115, 22)]
    for i, h in enumerate([60, 100, 140, 100, 60]):
        x = 70 + i * 30
        y_start = 128 - h // 2
        y_end = 128 + h // 2
        draw.line([x, y_start, x, y_end], fill=colors[i % 3], width=12)

    image.save('icon.ico', format='ICO')
    print("Icon created as icon.ico")

if __name__ == "__main__":
    create_icon()
