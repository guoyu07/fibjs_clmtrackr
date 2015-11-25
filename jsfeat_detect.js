// simple wrapper for jsfeat face detector
// requires jsfeat
var gd = require('gd');

var jsfeat = require("./lib/jsfeat.js");

var frontalface = require("./models/frontalface.js");

//debug
var drawPic = function(w, h, img_u8, name) {
	var image = gd.create(w, h);
	var pos1 = 0;
	for (var y = 0; y < h; y++)
		for (var x = 0; x < w; x++) {
			var pix = img_u8.data[pos1++];
			image.setPixel(x, y, gd.rgb(pix, pix, pix));
		}
	image.save(name + ".jpg");
}

var jsfeat_face = function(image) {

	var img_u8, ii_sum, ii_sqsum, ii_tilted;

	var w = image.width;
	var h = image.height;

	img_u8 = new jsfeat.matrix_t(w, h, jsfeat.U8_t | jsfeat.C1_t);

	ii_sum = new Int32Array((w + 1) * (h + 1));
	ii_sqsum = new Int32Array((w + 1) * (h + 1));
	ii_tilted = new Int32Array((w + 1) * (h + 1));

	var classifier = frontalface;

	this.findFace = function() {

		var imageData = {}
		for (var i = 0; i <= w * h * 4; i++)
			imageData[i] = 255;
		imageData["length"] = w * h * 4;

		var color, colorPatch
		var cursor = 0;

		for (var y = 0; y < h; y++) {
			for (var x = 0; x < w; x++) {
				color = image.getTrueColorPixel(x, y).toString(16)

				if (color.length < 6)
					colorPatch = new Array(6 - color.length + 1).join('0') + color
				else
					colorPatch = color

				imageData[cursor] = parseInt(colorPatch.substring(0, 2), 16)
				imageData[cursor + 1] = parseInt(colorPatch.substring(2, 4), 16)
				imageData[cursor + 2] = parseInt(colorPatch.substring(4, 6), 16)

				cursor += 4;
			}
		}

		jsfeat.imgproc.grayscale(imageData, img_u8.data);

		//debug
		drawPic(w, h, img_u8, "test_gray")

		// image.filter(gd.GRAYSCALE);
		// image.save("gray.jpg");

		jsfeat.imgproc.equalize_histogram(img_u8, img_u8);

		//debug
		drawPic(w, h, img_u8, "test_equalize")

		jsfeat.imgproc.compute_integral_image(img_u8, ii_sum, ii_sqsum, null);

		var rects = jsfeat.haar.detect_multi_scale(ii_sum, ii_sqsum, ii_tilted, null, img_u8.cols, img_u8.rows, classifier, 1.15, 2);

		rects = jsfeat.haar.group_rectangles(rects, 1);

		var rl = rects.length;

		if (rl > 0) {
			var best = rects[0];
			for (var i = 1; i < rl; i++) {
				if (rects[i].neighbors > best.neighbors) {
					best = rects[i]
				} else if (rects[i].neighbors == best.neighbors) {
					if (rects[i].confidence > best.confidence) best = rects[i];
				}
			}
			return [best];
		} else {
			return false;
		}
	}
}


module.exports = function(image) {
	return new jsfeat_face(image);
}