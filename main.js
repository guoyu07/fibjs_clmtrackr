"use strict";
//requires: ccv.js, numeric.js

var gd = require('gd'),
	process = require('process');

var pModel = require("./models/model_pca_20_svm.js")["pModel"];
var ctracker = require("./clm.js")();

// var image = gd.load("./media/franck_02221.jpg");
// var image = gd.load("./media/lyf.jpg");
// var image = gd.load("./media/cage.jpg");
// var image = gd.load("./media/audrey.jpg");
// var image = gd.load("./media/joconde.jpg");
// var image = gd.load("./media/wxl.png");
var image = gd.load("./media/dog.jpg");

ctracker.init(pModel);
ctracker.getInitialPosition(image)

// ctracker.track(image);
// var positions = ctracker.getCurrentPosition();
// console.error(positions)