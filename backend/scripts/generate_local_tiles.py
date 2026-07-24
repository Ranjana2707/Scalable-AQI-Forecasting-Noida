import os
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)

def num2deg(xtile, ytile, zoom):
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)

def generate_noida_tile(z, x, y, output_dir):
    tile_size = 256
    img = Image.new("RGBA", (tile_size, tile_size), (11, 15, 23, 255))
    draw = ImageDraw.Draw(img)
    
    nw_lat, nw_lon = num2deg(x, y, z)
    se_lat, se_lon = num2deg(x + 1, y + 1, z)
    
    # 1. Subtle Tile Grid Border
    draw.rectangle([0, 0, 255, 255], outline=(20, 28, 42, 255), width=1)
    
    # 2. Minor Coordinate Grid Lines
    for i in range(1, 4):
        offset = i * 64
        draw.line([(offset, 0), (offset, 256)], fill=(16, 23, 35, 255), width=1)
        draw.line([(0, offset), (256, offset)], fill=(16, 23, 35, 255), width=1)

    # Convert Lat/Lon to Tile Pixel
    def to_pix(lat, lon):
        px = int((lon - nw_lon) / (se_lon - nw_lon) * 256)
        py = int((nw_lat - lat) / (nw_lat - se_lat) * 256)
        return (px, py)

    # 3. Yamuna River (Blue-cyan accent curve)
    yamuna_pts = [
        (28.650, 77.290), (28.610, 77.310), (28.570, 77.320),
        (28.530, 77.330), (28.480, 77.360), (28.430, 77.400)
    ]
    yamuna_pix = [to_pix(lat, lon) for lat, lon in yamuna_pts]
    draw.line(yamuna_pix, fill=(14, 55, 80, 200), width=6)
    draw.line(yamuna_pix, fill=(34, 211, 238, 120), width=2)

    # 4. Hindon River
    hindon_pts = [
        (28.660, 77.380), (28.620, 77.400), (28.580, 77.420), (28.520, 77.450)
    ]
    hindon_pix = [to_pix(lat, lon) for lat, lon in hindon_pts]
    draw.line(hindon_pix, fill=(14, 55, 80, 160), width=4)

    # 5. Major Expressways (Noida Expressway & DND Flyway)
    exp_pts = [
        (28.580, 77.315), (28.545, 77.330), (28.500, 77.370), (28.460, 77.480)
    ]
    exp_pix = [to_pix(lat, lon) for lat, lon in exp_pts]
    draw.line(exp_pix, fill=(51, 65, 85, 255), width=3)
    draw.line(exp_pix, fill=(71, 85, 105, 180), width=1)

    # DND Flyway
    dnd_pts = [(28.586, 77.300), (28.580, 77.330)]
    dnd_pix = [to_pix(lat, lon) for lat, lon in dnd_pts]
    draw.line(dnd_pix, fill=(71, 85, 105, 200), width=2)

    # 6. Sector Urban Cluster Overlay Points (Sector 62, 1, 125, KP3)
    clusters = [
        (28.6244, 77.3789, "Sec-62"),
        (28.5844, 77.3159, "Sec-1"),
        (28.5456, 77.3261, "Sec-125"),
        (28.4682, 77.4912, "KP-III")
    ]
    for c_lat, c_lon, label in clusters:
        cx, cy = to_pix(c_lat, c_lon)
        if -30 <= cx <= 286 and -30 <= cy <= 286:
            draw.ellipse([cx-12, cy-12, cx+12, cy+12], outline=(30, 41, 59, 150), width=1)
            draw.ellipse([cx-4, cy-4, cx+4, cy+4], fill=(30, 41, 59, 200))

    # Save to disk
    tile_dir = output_dir / str(z) / str(x)
    tile_dir.mkdir(parents=True, exist_ok=True)
    img.save(tile_dir / f"{y}.png", "PNG")

def main():
    base_dir = Path(__file__).resolve().parents[1]
    output_dir = base_dir / "data/tiles"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("[TileGenerator] Generating local Noida dark GIS tiles...")
    # Bounding box for Noida region: Lat 28.4 to 28.7, Lon 77.2 to 77.6
    min_lat, max_lat = 28.40, 28.70
    min_lon, max_lon = 77.25, 77.55
    
    total = 0
    for z in range(8, 15):
        x1, y1 = deg2num(max_lat, min_lon, z)
        x2, y2 = deg2num(min_lat, max_lon, z)
        
        for x in range(min(x1, x2), max(x1, x2) + 1):
            for y in range(min(y1, y2), max(y1, y2) + 1):
                generate_noida_tile(z, x, y, output_dir)
                total += 1
                
    print(f"[TileGenerator] Successfully generated {total} local dark GIS tiles for Noida region!")

if __name__ == "__main__":
    main()
