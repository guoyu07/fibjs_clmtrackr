"use strict";

var gd = require('gd'),
	process = require('process');

var numeric = require("./lib/numeric.js");

var left_eye_filter = require("./models/left_eye_filter.js"),
	right_eye_filter = require("./models/right_eye_filter.js"),
	nose_filter = require("./models/nose_filter.js");

var mosseFilter = require("./filter/mosse.js"),
	svmFilter = require("./filter/svmfilter_fft.js");

var jsfeat_face = require("./jsfeat_detect.js");

function clm(params) {

	if (!params) params = {};
	if (params.constantVelocity === undefined) params.constantVelocity = true;
	if (params.searchWindow === undefined) params.searchWindow = 11;
	if (params.useWebGL === undefined) params.useWebGL = true;
	if (params.scoreThreshold === undefined) params.scoreThreshold = 0.5;
	if (params.stopOnConvergence === undefined) params.stopOnConvergence = false;
	if (params.weightPoints === undefined) params.weightPoints = undefined;
	if (params.sharpenResponse === undefined) params.sharpenResponse = false;

	var numPatches, patchSize, numParameters, patchType;
	var gaussianPD;
	var eigenVectors, eigenValues;
	var sketchCC, sketchW, sketchH, sketchCanvas;
	var candidate;
	var weights, model, biases;

	var sobelInit = false;
	var lbpInit = false;

	var currentParameters = [];
	var currentPositions = [];
	var previousParameters = [];
	var previousPositions = [];

	var patches = [];
	var responses = [];
	var meanShape = [];

	var responseMode = 'single';
	var responseList = ['raw'];
	var responseIndex = 0;

	/*
	It's possible to experiment with the sequence of variances used for the finding the maximum in the KDE.
	This sequence is pretty arbitrary, but was found to be okay using some manual testing.
	*/
	var varianceSeq = [10, 5, 1];
	//var varianceSeq = [3,1.5,0.75];
	//var varianceSeq = [6,3,0.75];
	var PDMVariance = 0.7;

	var relaxation = 0.1;

	var first = true;

	var convergenceLimit = 0.01;

	var learningRate = [];
	var stepParameter = 1.25;
	var prevCostFunc = []

	var searchWindow;
	var modelWidth, modelHeight;
	var halfSearchWindow, vecProbs, responsePixels;

	if (typeof Float64Array !== 'undefined') {
		var updatePosition = new Float64Array(2);
		var vecpos = new Float64Array(2);
	} else {
		var updatePosition = new Array(2);
		var vecpos = new Array(2);
	}
	var pw, pl, pdataLength;

	var facecheck_count = 0;

	var webglFi, svmFi, mosseCalc;

	// var scoringCanvas = document.createElement('canvas');
	// var scoringContext = scoringCanvas.getContext('2d');
	var msxmin, msymin, msxmax, msymax;
	var msmodelwidth, msmodelheight;
	var scoringWeights, scoringBias;
	var scoringHistory = [];
	var meanscore = 0;

	var mossef_lefteye, mossef_righteye, mossef_nose;
	var right_eye_position = [0.0, 0.0];
	var left_eye_position = [0.0, 0.0];
	var nose_position = [0.0, 0.0];
	var lep, rep, mep;
	var runnerTimeout, runnerElement, runnerBox;

	var pointWeights;

	var halfPI = Math.PI / 2;

	this.init = function(pdmmodel) {

		model = pdmmodel;

		// load from model
		patchType = model.patchModel.patchType;
		numPatches = model.patchModel.numPatches;
		patchSize = model.patchModel.patchSize[0];
		if (patchType == "MOSSE") {
			searchWindow = patchSize;
		} else {
			searchWindow = params.searchWindow;
		}
		numParameters = model.shapeModel.numEvalues;
		modelWidth = model.patchModel.canvasSize[0];
		modelHeight = model.patchModel.canvasSize[1];

		// set up canvas to work on
		// sketchCanvas = document.createElement('canvas');
		// sketchCC = sketchCanvas.getContext('2d');

		// sketchW = sketchCanvas.width = modelWidth + (searchWindow - 1) + patchSize - 1;
		// sketchH = sketchCanvas.height = modelHeight + (searchWindow - 1) + patchSize - 1;

		if (model.hints && mosseFilter && left_eye_filter && right_eye_filter && nose_filter) {
			//var mossef_lefteye = new mosseFilter({drawResponse : document.getElementById('overlay2')});
			mossef_lefteye = mosseFilter();
			mossef_lefteye.load(left_eye_filter);
			//var mossef_righteye = new mosseFilter({drawResponse : document.getElementById('overlay2')});
			mossef_righteye = mosseFilter();
			mossef_righteye.load(right_eye_filter);
			//var mossef_nose = new mosseFilter({drawResponse : document.getElementById('overlay2')});
			mossef_nose = mosseFilter();
			mossef_nose.load(nose_filter);
		} else {
			console.log("MOSSE filters not found, using rough approximation for initialization.");
		}

		// load eigenvectors
		eigenVectors = numeric.rep([numPatches * 2, numParameters], 0.0);
		for (var i = 0; i < numPatches * 2; i++) {
			for (var j = 0; j < numParameters; j++) {
				eigenVectors[i][j] = model.shapeModel.eigenVectors[i][j];
			}
		}

		// load mean shape
		for (var i = 0; i < numPatches; i++) {
			meanShape[i] = [model.shapeModel.meanShape[i][0], model.shapeModel.meanShape[i][1]];
		}

		// get max and mins, width and height of meanshape
		msxmax = msymax = 0;
		msxmin = msymin = 1000000;
		for (var i = 0; i < numPatches; i++) {
			if (meanShape[i][0] < msxmin) msxmin = meanShape[i][0];
			if (meanShape[i][1] < msymin) msymin = meanShape[i][1];
			if (meanShape[i][0] > msxmax) msxmax = meanShape[i][0];
			if (meanShape[i][1] > msymax) msymax = meanShape[i][1];
		}
		msmodelwidth = msxmax - msxmin;
		msmodelheight = msymax - msymin;

		// get scoringweights if they exist
		if (model.scoring) {
			scoringWeights = new Float64Array(model.scoring.coef);
			scoringBias = model.scoring.bias;
			// scoringCanvas.width = model.scoring.size[0];
			// scoringCanvas.height = model.scoring.size[1];
		}

		// load eigenvalues
		eigenValues = model.shapeModel.eigenValues;

		weights = model.patchModel.weights;
		biases = model.patchModel.bias;

		// precalculate gaussianPriorDiagonal
		gaussianPD = numeric.rep([numParameters + 4, numParameters + 4], 0);
		// set values and append manual inverse
		for (var i = 0; i < numParameters; i++) {
			if (model.shapeModel.nonRegularizedVectors.indexOf(i) >= 0) {
				gaussianPD[i + 4][i + 4] = 1 / 10000000;
			} else {
				gaussianPD[i + 4][i + 4] = 1 / eigenValues[i];
			}
		}

		for (var i = 0; i < numParameters + 4; i++) {
			currentParameters[i] = 0;
		}

		if (patchType == "SVM") {
			// var webGLContext;
			// var webGLTestCanvas = document.createElement('canvas');
			// if (window.WebGLRenderingContext) {
			// 	webGLContext = webGLTestCanvas.getContext('webgl') || webGLTestCanvas.getContext('experimental-webgl');
			// 	if (!webGLContext || !webGLContext.getExtension('OES_texture_float')) {
			// 		webGLContext = null;
			// 	}
			// }

			// if (webGLContext && params.useWebGL && (typeof(webglFilter) !== "undefined")) {
			// 	webglFi = new webglFilter();
			// 	try {
			// 		webglFi.init(weights, biases, numPatches, searchWindow + patchSize - 1, searchWindow + patchSize - 1, patchSize, patchSize);
			// 		if ('lbp' in weights) lbpInit = true;
			// 		if ('sobel' in weights) sobelInit = true;
			// 	} catch (err) {
			// 		alert("There was a problem setting up webGL programs, falling back to slightly slower javascript version. :(");
			// 		webglFi = undefined;
			// 		svmFi = new svmFilter();
			// 		svmFi.init(weights['raw'], biases['raw'], numPatches, patchSize, searchWindow);
			// 	}
			// } else if (typeof(svmFilter) !== "undefined") {
			// 	// use fft convolution if no webGL is available
			// 	svmFi = svmFilter();
			// 	svmFi.init(weights['raw'], biases['raw'], numPatches, patchSize, searchWindow);
			// } else {
			// 	throw "Could not initiate filters, please make sure that svmfilter.js or svmfilter_conv_js.js is loaded."
			// }
			svmFi = svmFilter();
			svmFi.init(weights['raw'], biases['raw'], numPatches, patchSize, searchWindow);
		} else if (patchType == "MOSSE") {
			mosseCalc = new mosseFilterResponses();
			mosseCalc.init(weights, numPatches, patchSize, patchSize);
		}

		if (patchType == "SVM") {
			pw = pl = patchSize + searchWindow - 1;
		} else {
			pw = pl = searchWindow;
		}
		pdataLength = pw * pl;
		halfSearchWindow = (searchWindow - 1) / 2;
		responsePixels = searchWindow * searchWindow;
		if (typeof Float64Array !== 'undefined') {
			vecProbs = new Float64Array(responsePixels);
			for (var i = 0; i < numPatches; i++) {
				patches[i] = new Float64Array(pdataLength);
			}
		} else {
			vecProbs = new Array(responsePixels);
			for (var i = 0; i < numPatches; i++) {
				patches[i] = new Array(pdataLength);
			}
		}

		for (var i = 0; i < numPatches; i++) {
			learningRate[i] = 1.0;
			prevCostFunc[i] = 0.0;
		}

		if (params.weightPoints) {
			// weighting of points 
			pointWeights = [];
			for (var i = 0; i < numPatches; i++) {
				if (i in params.weightPoints) {
					pointWeights[(i * 2)] = params.weightPoints[i];
					pointWeights[(i * 2) + 1] = params.weightPoints[i];
				} else {
					pointWeights[(i * 2)] = 1;
					pointWeights[(i * 2) + 1] = 1;
				}
			}
			pointWeights = numeric.diag(pointWeights);
		}

	}

	// detect position of face on canvas/video element
	var detectPosition = function(el) {
		// var canvas = document.createElement('canvas');
		// canvas.width = el.width;
		// canvas.height = el.height;
		// var cc = canvas.getContext('2d');
		// cc.drawImage(el, 0, 0, el.width, el.height);

		var jf = jsfeat_face(el);
		var comp = jf.findFace();

		if (comp.length > 0) {
			candidate = comp[0];
		} else {
			return false;
		}

		for (var i = 1; i < comp.length; i++) {
			if (comp[i].confidence > candidate.confidence) {
				candidate = comp[i];
			}
		}

		return candidate;
	}
	this.detectPosition = detectPosition;

	// get initial starting point for model
	var getInitialPosition = function(element, box) {
		var translateX, translateY, scaling, rotation;

		//resample the image
		var imageRange = 500;
		if (element.height > imageRange)
			element = element.resample(Math.round(imageRange * element.width / element.height), imageRange)

		if (box) {
			candidate = {
				x: box[0],
				y: box[1],
				width: box[2],
				height: box[3]
			};
		} else {
			var det = detectPosition(element);
			if (!det) {
				// if no face found, stop.
				return false;
			}
		}

		//debug
		var image = element.clone();
		var red = image.colorAllocate(255, 0, 0);
		image.rectangle(det.x, det.y, det.x + det.width, det.y + det.height, red);
		image.save("test_ori.jpg");
		process.system("open test_ori.jpg")
		console.error("findFace:", candidate)

		if (model.hints && mosseFilter && left_eye_filter && right_eye_filter && nose_filter) {
			var noseFilterWidth = candidate.width * 4.5 / 10;
			var eyeFilterWidth = candidate.width * 6 / 10;

			// detect position of eyes and nose via mosse filter
			var nose_result = mossef_nose.track(element, Math.round(candidate.x + (candidate.width / 2) - (noseFilterWidth / 2)), Math.round(candidate.y + candidate.height * (5 / 8) - (noseFilterWidth / 2)), noseFilterWidth, noseFilterWidth, false);
			var right_result = mossef_righteye.track(element, Math.round(candidate.x + (candidate.width * 3 / 4) - (eyeFilterWidth / 2)), Math.round(candidate.y + candidate.height * (2 / 5) - (eyeFilterWidth / 2)), eyeFilterWidth, eyeFilterWidth, false);
			var left_result = mossef_lefteye.track(element, Math.round(candidate.x + (candidate.width / 4) - (eyeFilterWidth / 2)), Math.round(candidate.y + candidate.height * (2 / 5) - (eyeFilterWidth / 2)), eyeFilterWidth, eyeFilterWidth, false);
			right_eye_position[0] = Math.round(candidate.x + (candidate.width * 3 / 4) - (eyeFilterWidth / 2)) + right_result[0];
			right_eye_position[1] = Math.round(candidate.y + candidate.height * (2 / 5) - (eyeFilterWidth / 2)) + right_result[1];
			left_eye_position[0] = Math.round(candidate.x + (candidate.width / 4) - (eyeFilterWidth / 2)) + left_result[0];
			left_eye_position[1] = Math.round(candidate.y + candidate.height * (2 / 5) - (eyeFilterWidth / 2)) + left_result[1];
			nose_position[0] = Math.round(candidate.x + (candidate.width / 2) - (noseFilterWidth / 2)) + nose_result[0];
			nose_position[1] = Math.round(candidate.y + candidate.height * (5 / 8) - (noseFilterWidth / 2)) + nose_result[1];

			// get eye and nose positions of model
			var lep = model.hints.leftEye;
			var rep = model.hints.rightEye;
			var mep = model.hints.nose;

			// get scaling, rotation, etc. via procrustes analysis
			var procrustes_params = procrustes([left_eye_position, right_eye_position, nose_position], [lep, rep, mep]);
			translateX = procrustes_params[0];
			translateY = procrustes_params[1];
			scaling = procrustes_params[2];
			rotation = procrustes_params[3];

			currentParameters[0] = (scaling * Math.cos(rotation)) - 1;
			currentParameters[1] = (scaling * Math.sin(rotation));
			currentParameters[2] = translateX;
			currentParameters[3] = translateY;

			//this.draw(document.getElementById('overlay'), currentParameters);

		} else {
			scaling = candidate.width / modelheight;
			//var ccc = document.getElementById('overlay').getContext('2d');
			//ccc.strokeRect(candidate.x,candidate.y,candidate.width,candidate.height);
			translateX = candidate.x - (xmin * scaling) + 0.1 * candidate.width;
			translateY = candidate.y - (ymin * scaling) + 0.25 * candidate.height;
			currentParameters[0] = scaling - 1;
			currentParameters[2] = translateX;
			currentParameters[3] = translateY;
		}

		currentPositions = calculatePositions(currentParameters, true);

		return [scaling, rotation, translateX, translateY];
	}
	this.getInitialPosition = getInitialPosition;

	// calculate positions from parameters
	var calculatePositions = function(parameters, useTransforms) {
		var x, y, a, b;
		var numParameters = parameters.length;
		var positions = [];
		for (var i = 0; i < numPatches; i++) {
			x = meanShape[i][0];
			y = meanShape[i][1];
			for (var j = 0; j < numParameters - 4; j++) {
				x += model.shapeModel.eigenVectors[(i * 2)][j] * parameters[j + 4];
				y += model.shapeModel.eigenVectors[(i * 2) + 1][j] * parameters[j + 4];
			}
			if (useTransforms) {
				a = parameters[0] * x - parameters[1] * y + parameters[2];
				b = parameters[0] * y + parameters[1] * x + parameters[3];
				x += a;
				y += b;
			}
			positions[i] = [x, y];
		}

		return positions;
	}

	// calculate score of current fit
	var checkTracking = function() {
		scoringContext.drawImage(sketchCanvas, Math.round(msxmin + (msmodelwidth / 4.5)), Math.round(msymin - (msmodelheight / 12)), Math.round(msmodelwidth - (msmodelwidth * 2 / 4.5)), Math.round(msmodelheight - (msmodelheight / 12)), 0, 0, 20, 22);
		// getImageData of canvas
		var imgData = scoringContext.getImageData(0, 0, 20, 22);
		// convert data to grayscale
		var scoringData = new Array(20 * 22);
		var scdata = imgData.data;
		var scmax = 0;
		for (var i = 0; i < 20 * 22; i++) {
			scoringData[i] = scdata[i * 4] * 0.3 + scdata[(i * 4) + 1] * 0.59 + scdata[(i * 4) + 2] * 0.11;
			scoringData[i] = Math.log(scoringData[i] + 1);
			if (scoringData[i] > scmax) scmax = scoringData[i];
		}

		if (scmax > 0) {
			// normalize & multiply by svmFilter
			var mean = 0;
			for (var i = 0; i < 20 * 22; i++) {
				mean += scoringData[i];
			}
			mean /= (20 * 22);
			var sd = 0;
			for (var i = 0; i < 20 * 22; i++) {
				sd += (scoringData[i] - mean) * (scoringData[i] - mean);
			}
			sd /= (20 * 22 - 1)
			sd = Math.sqrt(sd);

			var score = 0;
			for (var i = 0; i < 20 * 22; i++) {
				scoringData[i] = (scoringData[i] - mean) / sd;
				score += (scoringData[i]) * scoringWeights[i];
			}
			score += scoringBias;
			score = 1 / (1 + Math.exp(-score));

			scoringHistory.splice(0, scoringHistory.length == 5 ? 1 : 0);
			scoringHistory.push(score);

			if (scoringHistory.length > 4) {
				// get average
				meanscore = 0;
				for (var i = 0; i < 5; i++) {
					meanscore += scoringHistory[i];
				}
				meanscore /= 5;
				// if below threshold, then reset (return false)
				if (meanscore < params.scoreThreshold) return false;
			}
		}
		return true;
	}

	this.track = function(element, box) {

		var scaling, translateX, translateY, rotation;
		var croppedPatches = [];
		var ptch, px, py;

		if (first) {
			// do viola-jones on canvas to get initial guess, if we don't have any points
			var gi = getInitialPosition(element, box);
			// if (!gi) {
			// 	// send an event on no face found
			// 	var evt = document.createEvent("Event");
			// 	evt.initEvent("clmtrackrNotFound", true, true);
			// 	document.dispatchEvent(evt)

			// 	return false;
			// }
			scaling = gi[0];
			rotation = gi[1];
			translateX = gi[2];
			translateY = gi[3];

			first = false;
		} else {
			facecheck_count += 1;

			if (params.constantVelocity) {
				// calculate where to get patches via constant velocity prediction
				if (previousParameters.length >= 2) {
					for (var i = 0; i < currentParameters.length; i++) {
						currentParameters[i] = (relaxation) * previousParameters[1][i] + (1 - relaxation) * ((2 * previousParameters[1][i]) - previousParameters[0][i]);
						//currentParameters[i] = (3*previousParameters[2][i]) - (3*previousParameters[1][i]) + previousParameters[0][i];
					}
				}
			}

			// change translation, rotation and scale parameters
			rotation = halfPI - Math.atan((currentParameters[0] + 1) / currentParameters[1]);
			if (rotation > halfPI) {
				rotation -= Math.PI;
			}
			scaling = currentParameters[1] / Math.sin(rotation);
			translateX = currentParameters[2];
			translateY = currentParameters[3];
		}

		// copy canvas to a new dirty canvas
		// sketchCC.save();

		// // clear canvas
		// sketchCC.clearRect(0, 0, sketchW, sketchH);

		// sketchCC.scale(1 / scaling, 1 / scaling);
		// sketchCC.rotate(-rotation);
		// sketchCC.translate(-translateX, -translateY);

		// sketchCC.drawImage(element, 0, 0, element.width, element.height);

		// sketchCC.restore();
		//	get cropped images around new points based on model parameters (not scaled and translated)
		var patchPositions = calculatePositions(currentParameters, false);

		// check whether tracking is ok
		if (scoringWeights && (facecheck_count % 10 == 0)) {
			if (!checkTracking()) {
				// reset all parameters
				first = true;
				scoringHistory = [];
				for (var i = 0; i < currentParameters.length; i++) {
					currentParameters[i] = 0;
					previousParameters = [];
				}

				// send event to signal that tracking was lost
				var evt = document.createEvent("Event");
				evt.initEvent("clmtrackrLost", true, true);
				document.dispatchEvent(evt)

				return false;
			}
		}


		var pdata, pmatrix, grayscaleColor;
		for (var i = 0; i < numPatches; i++) {
			px = patchPositions[i][0] - (pw / 2);
			py = patchPositions[i][1] - (pl / 2);
			ptch = sketchCC.getImageData(Math.round(px), Math.round(py), pw, pl);
			pdata = ptch.data;

			// convert to grayscale
			pmatrix = patches[i];
			for (var j = 0; j < pdataLength; j++) {
				grayscaleColor = pdata[j * 4] * 0.3 + pdata[(j * 4) + 1] * 0.59 + pdata[(j * 4) + 2] * 0.11;
				pmatrix[j] = grayscaleColor;
			}
		}

		/*print weights*/
		/*sketchCC.clearRect(0, 0, sketchW, sketchH);
		var nuWeights;
		for (var i = 0;i < numPatches;i++) {
			nuWeights = weights[i].map(function(x) {return x*2000+127;});
			drawData(sketchCC, nuWeights, patchSize, patchSize, false, patchPositions[i][0]-(patchSize/2), patchPositions[i][1]-(patchSize/2));
		}*/

		// print patches
		/*sketchCC.clearRect(0, 0, sketchW, sketchH);
		for (var i = 0;i < numPatches;i++) {
			if ([27,32,44,50].indexOf(i) > -1) {
				drawData(sketchCC, patches[i], pw, pl, false, patchPositions[i][0]-(pw/2), patchPositions[i][1]-(pl/2));
			}
		}*/
		if (patchType == "SVM") {
			if (typeof(webglFi) !== "undefined") {
				responses = getWebGLResponses(patches);
			} else if (typeof(svmFi) !== "undefined") {
				responses = svmFi.getResponses(patches);
			} else {
				throw "SVM-filters do not seem to be initiated properly."
			}
		} else if (patchType == "MOSSE") {
			responses = mosseCalc.getResponses(patches);
		}

		// option to increase sharpness of responses
		if (params.sharpenResponse) {
			for (var i = 0; i < numPatches; i++) {
				for (var j = 0; j < responses[i].length; j++) {
					responses[i][j] = Math.pow(responses[i][j], params.sharpenResponse);
				}
			}
		}

		// print responses
		/*sketchCC.clearRect(0, 0, sketchW, sketchH);
		var nuWeights;
		for (var i = 0;i < numPatches;i++) {
		
			nuWeights = [];
			for (var j = 0;j < responses[i].length;j++) {
				nuWeights.push(responses[i][j]*255);
			}
			
			//if ([27,32,44,50].indexOf(i) > -1) {
			//	drawData(sketchCC, nuWeights, searchWindow, searchWindow, false, patchPositions[i][0]-((searchWindow-1)/2), patchPositions[i][1]-((searchWindow-1)/2));
			//}
			drawData(sketchCC, nuWeights, searchWindow, searchWindow, false, patchPositions[i][0]-((searchWindow-1)/2), patchPositions[i][1]-((searchWindow-1)/2));
		}*/

		// iterate until convergence or max 10, 20 iterations?:
		var originalPositions = currentPositions;
		var jac;
		var meanshiftVectors = [];

		for (var i = 0; i < varianceSeq.length; i++) {

			// calculate jacobian
			jac = createJacobian(currentParameters, eigenVectors);

			// for debugging
			//var debugMVs = [];
			//

			var opj0, opj1;

			for (var j = 0; j < numPatches; j++) {
				opj0 = originalPositions[j][0] - ((searchWindow - 1) * scaling / 2);
				opj1 = originalPositions[j][1] - ((searchWindow - 1) * scaling / 2);

				// calculate PI x gaussians
				var vpsum = gpopt(searchWindow, currentPositions[j], updatePosition, vecProbs, responses, opj0, opj1, j, varianceSeq[i], scaling);

				// calculate meanshift-vector
				gpopt2(searchWindow, vecpos, updatePosition, vecProbs, vpsum, opj0, opj1, scaling);

				// for debugging
				//var debugMatrixMV = gpopt2(searchWindow, vecpos, updatePosition, vecProbs, vpsum, opj0, opj1);

				// evaluate here whether to increase/decrease stepSize
				/*if (vpsum >= prevCostFunc[j]) {
					learningRate[j] *= stepParameter;
				} else {
					learningRate[j] = 1.0;
				}
				prevCostFunc[j] = vpsum;*/

				// compute mean shift vectors
				// extrapolate meanshiftvectors
				/*var msv = [];
				msv[0] = learningRate[j]*(vecpos[0] - currentPositions[j][0]);
				msv[1] = learningRate[j]*(vecpos[1] - currentPositions[j][1]);
				meanshiftVectors[j] = msv;*/
				meanshiftVectors[j] = [vecpos[0] - currentPositions[j][0], vecpos[1] - currentPositions[j][1]];

				//if (isNaN(msv[0]) || isNaN(msv[1])) debugger;

				//for debugging
				//debugMVs[j] = debugMatrixMV;
				//
			}

			// draw meanshiftVector
			/*sketchCC.clearRect(0, 0, sketchW, sketchH);
			var nuWeights;
			for (var npidx = 0;npidx < numPatches;npidx++) {
				nuWeights = debugMVs[npidx].map(function(x) {return x*255*500;});
				drawData(sketchCC, nuWeights, searchWindow, searchWindow, false, patchPositions[npidx][0]-((searchWindow-1)/2), patchPositions[npidx][1]-((searchWindow-1)/2));
			}*/

			var meanShiftVector = numeric.rep([numPatches * 2, 1], 0.0);
			for (var k = 0; k < numPatches; k++) {
				meanShiftVector[k * 2][0] = meanshiftVectors[k][0];
				meanShiftVector[(k * 2) + 1][0] = meanshiftVectors[k][1];
			}

			// compute pdm parameter update
			//var prior = numeric.mul(gaussianPD, PDMVariance);
			var prior = numeric.mul(gaussianPD, varianceSeq[i]);
			if (params.weightPoints) {
				var jtj = numeric.dot(numeric.transpose(jac), numeric.dot(pointWeights, jac));
			} else {
				var jtj = numeric.dot(numeric.transpose(jac), jac);
			}
			var cpMatrix = numeric.rep([numParameters + 4, 1], 0.0);
			for (var l = 0; l < (numParameters + 4); l++) {
				cpMatrix[l][0] = currentParameters[l];
			}
			var priorP = numeric.dot(prior, cpMatrix);
			if (params.weightPoints) {
				var jtv = numeric.dot(numeric.transpose(jac), numeric.dot(pointWeights, meanShiftVector));
			} else {
				var jtv = numeric.dot(numeric.transpose(jac), meanShiftVector);
			}
			var paramUpdateLeft = numeric.add(prior, jtj);
			var paramUpdateRight = numeric.sub(priorP, jtv);
			var paramUpdate = numeric.dot(numeric.inv(paramUpdateLeft), paramUpdateRight);
			//var paramUpdate = numeric.solve(paramUpdateLeft, paramUpdateRight, true);

			var oldPositions = currentPositions;

			// update estimated parameters
			for (var k = 0; k < numParameters + 4; k++) {
				currentParameters[k] -= paramUpdate[k];
			}

			// clipping of parameters if they're too high
			var clip;
			for (var k = 0; k < numParameters; k++) {
				clip = Math.abs(3 * Math.sqrt(eigenValues[k]));
				if (Math.abs(currentParameters[k + 4]) > clip) {
					if (currentParameters[k + 4] > 0) {
						currentParameters[k + 4] = clip;
					} else {
						currentParameters[k + 4] = -clip;
					}
				}

			}

			// update current coordinates
			currentPositions = calculatePositions(currentParameters, true);

			// check if converged
			// calculate norm of parameterdifference
			var positionNorm = 0;
			var pnsq_x, pnsq_y;
			for (var k = 0; k < currentPositions.length; k++) {
				pnsq_x = (currentPositions[k][0] - oldPositions[k][0]);
				pnsq_y = (currentPositions[k][1] - oldPositions[k][1]);
				positionNorm += ((pnsq_x * pnsq_x) + (pnsq_y * pnsq_y));
			}
			//console.log("positionnorm:"+positionNorm);

			// if norm < limit, then break
			if (positionNorm < convergenceLimit) {
				break;
			}

		}

		if (params.constantVelocity) {
			// add current parameter to array of previous parameters
			previousParameters.push(currentParameters.slice());
			previousParameters.splice(0, previousParameters.length == 3 ? 1 : 0);
		}

		// store positions, for checking convergence
		previousPositions.splice(0, previousPositions.length == 10 ? 1 : 0);
		previousPositions.push(currentPositions.slice(0));

		// send an event on each iteration
		var evt = document.createEvent("Event");
		evt.initEvent("clmtrackrIteration", true, true);
		document.dispatchEvent(evt)

		if (this.getConvergence() < 0.5) {
			// we must get a score before we can say we've converged
			if (scoringHistory.length >= 5) {
				if (params.stopOnConvergence) {
					this.stop();
				}

				var evt = document.createEvent("Event");
				evt.initEvent("clmtrackrConverged", true, true);
				document.dispatchEvent(evt)
			}
		}

		// return new points
		return currentPositions;
	}


	/*
	 *	get coordinates of current model fit
	 */
	this.getCurrentPosition = function() {
		if (first) {
			return false;
		} else {
			return currentPositions;
		}
	}
}

module.exports = function() {
	return new clm();
}