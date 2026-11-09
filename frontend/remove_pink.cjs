const jimp = require('jimp');
const fs = require('fs');

async function processImage(filename) {
    try {
        let image;
        if (typeof jimp.read === 'function') {
            image = await jimp.read(filename);
        } else if (jimp.Jimp && typeof jimp.Jimp.read === 'function') {
            image = await jimp.Jimp.read(filename);
        } else {
            console.error("Could not find Jimp.read function.");
            return;
        }
        
        const bgColorInt = image.getPixelColor(0, 0);
        let bgRgba;
        if (typeof jimp.intToRGBA === 'function') {
            bgRgba = jimp.intToRGBA(bgColorInt);
        } else if (jimp.Jimp && typeof jimp.Jimp.intToRGBA === 'function') {
            bgRgba = jimp.Jimp.intToRGBA(bgColorInt);
        } else {
            bgRgba = {
                r: (bgColorInt >> 24) & 255,
                g: (bgColorInt >> 16) & 255,
                b: (bgColorInt >> 8) & 255,
                a: bgColorInt & 255
            };
        }
        
        console.log(`Processing ${filename}. Background color found: rgb(${bgRgba.r}, ${bgRgba.g}, ${bgRgba.b})`);
        
        const colorDistance = (c1, c2) => {
            return Math.sqrt(Math.pow(c1.r - c2.r, 2) + Math.pow(c1.g - c2.g, 2) + Math.pow(c1.b - c2.b, 2));
        };
        
        const tolerance = 45; 
        
        image.scan(0, 0, image.bitmap.width, image.bitmap.height, function(x, y, idx) {
            const r = this.bitmap.data[idx + 0];
            const g = this.bitmap.data[idx + 1];
            const b = this.bitmap.data[idx + 2];
            
            const dist = colorDistance(bgRgba, { r, g, b });
            
            if (dist < tolerance) {
                this.bitmap.data[idx + 3] = 0; // Set alpha to 0
            }
        });
        
        if (typeof image.writeAsync === 'function') {
            await image.writeAsync(filename);
        } else if (typeof image.write === 'function') {
            // jimp v1
            image.write(filename);
        }
        
        console.log(`Successfully made background transparent for ${filename}`);
    } catch (e) {
        console.error(`Error processing ${filename}:`, e);
    }
}

async function main() {
    console.log("Starting image background removal...");
    await processImage('src/assets/outfit_1.png');
    await processImage('src/assets/outfit_2.png');
    console.log("Done!");
}

main();
