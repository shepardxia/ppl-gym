var bins = map(function(i) { return i * 0.2; }, _.range(30));

var gaussianUnnorm = function(x, mu, sigma) {
  return Math.exp(-0.5 * Math.pow((x - mu) / sigma, 2));
};

var classParams = {
  child: {mu: 0.5, sigma: 1},
  adult: {mu: 2, sigma: 3},
  bodybuilder: {mu: 5, sigma: 3}
};

var superParams = {mu: 3, sigma: 1};

var getStateDist = function(compClass) {
  var params = compClass === 'superordinate' ? superParams : classParams[compClass];
  var weights = map(function(x) { return gaussianUnnorm(x, params.mu, params.sigma); }, bins);
  var totalWeight = sum(weights);
  var probs = map(function(w) { return w / totalWeight; }, weights);
  return Categorical({vs: bins, ps: probs});
};

var utterances = ["heavy", "light"];

var meaning = function(utt, state, threshold) {
  if (utt === "heavy") {
    return state > threshold ? 0.9999 : 0.0001;
  } else {
    return state < threshold ? 0.9999 : 0.0001;
  }
};

var getThresholdForUtt = function(utt, x) {
  if (utt === "heavy") { return x - 0.1; }
  else { return x + 0.1; }
};

var alpha = 5;

// Literal listener: given utterance, threshold, compClass -> distribution over state
var literalListener = mem(function(utt, threshold, compClass) {
  return Infer({method: 'enumerate'}, function() {
    var state = sample(getStateDist(compClass));
    var prob = meaning(utt, state, threshold);
    factor(Math.log(prob));
    return state;
  });
});

// Speaker: given state, threshold, compClass -> distribution over utterance
var speaker = mem(function(state, threshold, compClass) {
  return Infer({method: 'enumerate'}, function() {
    var utt = uniformDraw(utterances);
    var ll = literalListener(utt, threshold, compClass);
    factor(alpha * ll.score(state));
    return utt;
  });
});

// Comparison class prior given speaker identity
var compClassPrior = function(speakerIdentity) {
  var classes = ['child', 'adult', 'bodybuilder'];
  var probs = speakerIdentity === 'child' ? [0.75, 0.25, 0.15] :
              speakerIdentity === 'adult' ? [0.01, 0.70, 0.50] :
              [0.0001, 0.20, 0.99];
  var total = sum(probs);
  var normProbs = map(function(p) { return p / total; }, probs);
  return Categorical({vs: classes, ps: normProbs});
};

// Pragmatic listener: hears utterance, knows speaker identity
var pragmaticListener = function(utt, speakerIdentity) {
  return Infer({method: 'enumerate'}, function() {
    // Sample comparison class from prior given speaker identity
    var compClass = sample(compClassPrior(speakerIdentity));
    // Sample state from that class's distribution
    var state = sample(getStateDist(compClass));
    // Sample threshold uniformly from bins
    var threshold = uniformDraw(bins);
    // Condition on the speaker choosing this utterance
    var speakerDist = speaker(state, threshold, compClass);
    observe(speakerDist, utt);
    return state;
  });
};

var ANSWER = pragmaticListener("heavy", "child");