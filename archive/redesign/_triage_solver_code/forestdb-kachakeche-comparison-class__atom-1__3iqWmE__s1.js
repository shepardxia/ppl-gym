var bins = map(function(i) { return i * 0.2; }, _.range(30));

var gaussianDensity = function(x, mu, sigma) {
  return Math.exp(-0.5 * Math.pow((x - mu) / sigma, 2)) / (sigma * Math.sqrt(2 * Math.PI));
};

var classParams = {
  child: {mu: 0.5, sigma: 1},
  adult: {mu: 2, sigma: 3},
  bodybuilder: {mu: 5, sigma: 3}
};

var superParams = {mu: 3, sigma: 1};

var getWeights = function(params) {
  var densities = map(function(x) { return gaussianDensity(x, params.mu, params.sigma); }, bins);
  var total = sum(densities);
  return map(function(d) { return d / total; }, densities);
};

var classWeights = {
  child: getWeights(classParams.child),
  adult: getWeights(classParams.adult),
  bodybuilder: getWeights(classParams.bodybuilder)
};

var utterances = ["heavy", "light"];

var meaning = function(utt, state, threshold) {
  if (utt === "heavy") {
    return state > threshold ? 0.9999 : 0.0001;
  } else {
    return state < threshold ? 0.9999 : 0.0001;
  }
};

var getThreshold = function(utt, x) {
  if (utt === "heavy") { return x - 0.1; }
  else { return x + 0.1; }
};

var alpha = 5;

var literalListener = mem(function(utt, threshold, compClass) {
  return Infer({method: 'enumerate'}, function() {
    var weights = classWeights[compClass];
    var state = categorical({vs: bins, ps: weights});
    factor(Math.log(meaning(utt, state, threshold)));
    return state;
  });
});

var speaker = mem(function(state, threshold, compClass) {
  return Infer({method: 'enumerate'}, function() {
    var utt = uniformDraw(utterances);
    var ll = literalListener(utt, threshold, compClass);
    factor(alpha * ll.score(state));
    return utt;
  });
});

var ccPriors = {
  child:       {vs: ['child', 'adult', 'bodybuilder'], ps: [0.75, 0.25, 0.15]},
  adult:       {vs: ['child', 'adult', 'bodybuilder'], ps: [0.01, 0.70, 0.50]},
  bodybuilder: {vs: ['child', 'adult', 'bodybuilder'], ps: [0.0001, 0.20, 0.99]}
};

var normalizePrior = function(speakerIdentity) {
  var prior = ccPriors[speakerIdentity];
  var total = sum(prior.ps);
  return {vs: prior.vs, ps: map(function(p) { return p / total; }, prior.ps)};
};

var pragmaticListener = function(utt, speakerIdentity) {
  return Infer({method: 'enumerate'}, function() {
    var normPrior = normalizePrior(speakerIdentity);
    var compClass = categorical({vs: normPrior.vs, ps: normPrior.ps});
    var weights = classWeights[compClass];
    var state = categorical({vs: bins, ps: weights});
    var threshold = uniformDraw(bins);
    var sp = speaker(state, threshold, compClass);
    observe(sp, utt);
    return state;
  });
};

var ANSWER = pragmaticListener("heavy", "child");