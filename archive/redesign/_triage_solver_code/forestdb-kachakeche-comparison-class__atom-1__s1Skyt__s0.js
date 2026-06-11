var bins = _.range(30).map(function(i) { return Math.round(i * 0.2 * 10) / 10; });

var gaussianWeights = function(mu, sigma) {
  var raw = bins.map(function(x) {
    var diff = x - mu;
    return Math.exp(-0.5 * diff * diff / (sigma * sigma));
  });
  var total = raw.reduce(function(a, b) { return a + b; }, 0);
  return raw.map(function(w) { return w / total; });
};

var classParams = {
  child: {mu: 0.5, sigma: 1},
  adult: {mu: 2, sigma: 3},
  bodybuilder: {mu: 5, sigma: 3},
  superordinate: {mu: 3, sigma: 1}
};

var classWeights = {
  child: gaussianWeights(0.5, 1),
  adult: gaussianWeights(2, 3),
  bodybuilder: gaussianWeights(5, 3),
  superordinate: gaussianWeights(3, 1)
};

var classes = ["child", "adult", "bodybuilder"];

var ccPrior = {
  child: {vs: ["child", "adult", "bodybuilder"], ps: [0.75, 0.25, 0.15]},
  adult: {vs: ["child", "adult", "bodybuilder"], ps: [0.01, 0.70, 0.50]},
  bodybuilder: {vs: ["child", "adult", "bodybuilder"], ps: [0.0001, 0.20, 0.99]}
};

var normalizePrior = function(ps) {
  var total = ps.reduce(function(a, b) { return a + b; }, 0);
  return ps.map(function(p) { return p / total; });
};

var utterances = ["heavy", "light"];

var meaning = function(utt, state, threshold) {
  if (utt === "heavy") {
    return state > threshold ? 0.9999 : 0.0001;
  } else {
    return state < threshold ? 0.9999 : 0.0001;
  }
};

var thresholdForHeavy = function(x) { return x - 0.1; };
var thresholdForLight = function(x) { return x + 0.1; };

// Literal listener: given utterance, threshold (for that utterance), and comparison class,
// returns distribution over states
var literalListener = mem(function(utt, threshold, cc) {
  return Infer({method: 'enumerate'}, function() {
    var ws = classWeights[cc];
    var stateIdx = sample(Categorical({vs: _.range(30), ps: ws}));
    var state = bins[stateIdx];
    var m = meaning(utt, state, threshold);
    factor(Math.log(m));
    return state;
  });
});

// Speaker: given state, threshold, and comparison class, chooses utterance
var alpha = 5;

var speaker = mem(function(state, thresholdHeavy, thresholdLight, cc) {
  return Infer({method: 'enumerate'}, function() {
    var utt = utterances[randomInteger(2)];
    var threshold = utt === "heavy" ? thresholdHeavy : thresholdLight;
    var ll = utt === "heavy" ? literalListener("heavy", threshold, cc) : literalListener("light", threshold, cc);
    var score = ll.score(state);
    factor(alpha * score);
    return utt;
  });
});

// Pragmatic listener
var pragmaticListener = function(utt, speakerIdentity) {
  return Infer({method: 'enumerate'}, function() {
    // Sample comparison class from prior given speaker identity
    var prior = ccPrior[speakerIdentity];
    var normPs = normalizePrior(prior.ps);
    var cc = sample(Categorical({vs: prior.vs, ps: normPs}));
    
    // Sample state from comparison class distribution
    var ws = classWeights[cc];
    var stateIdx = sample(Categorical({vs: _.range(30), ps: ws}));
    var state = bins[stateIdx];
    
    // Sample thresholds uniformly from bins
    var thresholdHeavyIdx = randomInteger(30);
    var thresholdHeavy = thresholdForHeavy(bins[thresholdHeavyIdx]);
    
    var thresholdLightIdx = randomInteger(30);
    var thresholdLight = thresholdForLight(bins[thresholdLightIdx]);
    
    // Observe utterance from speaker
    var speakerDist = speaker(state, thresholdHeavy, thresholdLight, cc);
    observe(speakerDist, utt);
    
    return state;
  });
};

var ANSWER = pragmaticListener("heavy", "child");