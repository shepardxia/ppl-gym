var bins = _.range(0, 30).map(function(i) { return i * 0.2; });

var gaussianUnnorm = function(x, mu, sigma) {
  return Math.exp(-0.5 * Math.pow((x - mu) / sigma, 2));
};

var makeWeights = function(mu, sigma) {
  return bins.map(function(x) { return gaussianUnnorm(x, mu, sigma); });
};

var classParams = {
  child: {mu: 0.5, sigma: 1},
  adult: {mu: 2, sigma: 3},
  bodybuilder: {mu: 5, sigma: 3}
};

var superParams = {mu: 3, sigma: 1};

var classWeights = {
  child: makeWeights(0.5, 1),
  adult: makeWeights(2, 3),
  bodybuilder: makeWeights(5, 3)
};

var utterances = ["heavy", "light"];

var meaning = function(utt, state, threshold) {
  if (utt === "heavy") {
    return state > threshold ? 0.9999 : 0.0001;
  } else {
    return state < threshold ? 0.9999 : 0.0001;
  }
};

var thresholdForUtt = function(utt, x) {
  if (utt === "heavy") { return x - 0.1; }
  else { return x + 0.1; }
};

var alpha = 5;

// Literal listener: given utterance, threshold, comparison class
// returns distribution over states
var literalListener = mem(function(utt, threshold, cc) {
  return Infer({method: 'enumerate'}, function() {
    var weights = classWeights[cc];
    var stateIdx = sample(Categorical({vs: _.range(0, 30), ps: weights}));
    var state = bins[stateIdx];
    factor(Math.log(meaning(utt, state, threshold)));
    return state;
  });
});

// Speaker: given state, threshold, cc, returns distribution over utterances
var speaker = mem(function(state, threshold, cc) {
  return Infer({method: 'enumerate'}, function() {
    var utt = uniformDraw(utterances);
    var ll = literalListener(utt, threshold, cc);
    factor(alpha * ll.score(state));
    return utt;
  });
});

// Comparison class prior given speaker identity
var ccPriorFn = function(speakerIdentity) {
  if (speakerIdentity === "child") {
    return Categorical({vs: ["child", "adult", "bodybuilder"], ps: [0.75, 0.25, 0.15]});
  } else if (speakerIdentity === "adult") {
    return Categorical({vs: ["child", "adult", "bodybuilder"], ps: [0.01, 0.70, 0.50]});
  } else {
    return Categorical({vs: ["child", "adult", "bodybuilder"], ps: [0.0001, 0.20, 0.99]});
  }
};

// Pragmatic listener
var pragmaticListener = function(utt, speakerIdentity) {
  return Infer({method: 'enumerate'}, function() {
    // Sample comparison class from prior given speaker identity
    var cc = sample(ccPriorFn(speakerIdentity));
    
    // Sample state from comparison class distribution
    var weights = classWeights[cc];
    var stateIdx = sample(Categorical({vs: _.range(0, 30), ps: weights}));
    var state = bins[stateIdx];
    
    // Sample thresholds uniformly from bins
    var heavyThreshIdx = randomInteger(30);
    var lightThreshIdx = randomInteger(30);
    var heavyThresh = bins[heavyThreshIdx] - 0.1;
    var lightThresh = bins[lightThreshIdx] + 0.1;
    
    var threshold = utt === "heavy" ? heavyThresh : lightThresh;
    
    // Observe speaker's utterance
    var spkDist = speaker(state, threshold, cc);
    observe(spkDist, utt);
    
    return state;
  });
};

var ANSWER = pragmaticListener("heavy", "child");