var binParam = 3;
var alpha = 5;

var stateValues = map(function(i) { return -3 + i * (1/3); }, _.range(0, 18));

var gaussPDF = function(x, mu, sigma) {
  return Math.exp(-0.5 * Math.pow((x - mu) / sigma, 2)) / (sigma * Math.sqrt(2 * Math.PI));
};

var superParams = {mu: 0, sigma: 1};
var subParams = {mu: 1, sigma: 0.5};

var stateProbs = function(params) {
  return map(function(x) { return gaussPDF(x, params.mu, params.sigma); }, stateValues);
};

var superProbs = stateProbs(superParams);
var subProbs = stateProbs(subParams);

var statePrior = function(cc) {
  return Categorical({vs: stateValues, ps: (cc === "super") ? superProbs : subProbs});
};

var tallThresholds = map(function(x) { return x - 1/(2*binParam); }, stateValues);
var shortThresholds = map(function(x) { return x + 1/(2*binParam); }, stateValues);

var thresholdPrior = function(type) {
  return (type === "tall") ? uniformDraw(tallThresholds) : uniformDraw(shortThresholds);
};

var utterances = ["tall", "short", "silence"];

var meaning = function(utterance, state, tallThresh, shortThresh) {
  return (utterance === "tall") ? (state > tallThresh) :
         (utterance === "short") ? (state < shortThresh) :
         true;
};

var literalListener = cache(function(utterance, tallThresh, shortThresh, cc) {
  return Infer({method: "enumerate"}, function() {
    var state = sample(statePrior(cc));
    condition(meaning(utterance, state, tallThresh, shortThresh));
    return state;
  });
});

var speaker = cache(function(state, tallThresh, shortThresh, cc) {
  return Infer({method: "enumerate"}, function() {
    var utterance = uniformDraw(utterances);
    var L = literalListener(utterance, tallThresh, shortThresh, cc);
    factor(alpha * L.score(state));
    return utterance;
  });
});

var pragmaticListener = function(utterance, subParamsCC) {
  return Infer({method: "enumerate"}, function() {
    var cc = uniformDraw(["super", "sub"]);
    var state = sample(statePrior(subParamsCC));
    var tallThresh = thresholdPrior("tall");
    var shortThresh = thresholdPrior("short");
    var S = speaker(state, tallThresh, shortThresh, cc);
    observe(S, utterance);
    return cc;
  });
};

var ANSWER = pragmaticListener("tall", "sub");