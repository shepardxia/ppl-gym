var binParam = 3;

var stateValues = map(function(i) { return -3 + i * (1/3); }, _.range(18));

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

var stateDist = function(cc) {
  return Categorical({vs: stateValues, ps: cc === 'super' ? superProbs : subProbs});
};

var tallThresholds = map(function(x) { return x - 1/(2*binParam); }, stateValues);
var shortThresholds = map(function(x) { return x + 1/(2*binParam); }, stateValues);

var alpha = 5;

var meaning = function(utterance, state, tallThresh, shortThresh) {
  return utterance === 'tall' ? state > tallThresh :
         utterance === 'short' ? state < shortThresh :
         true;
};

var literalListener = cache(function(utterance, tallThresh, shortThresh, cc) {
  return Infer({method: 'enumerate'}, function() {
    var state = sample(stateDist(cc));
    condition(meaning(utterance, state, tallThresh, shortThresh));
    return state;
  });
});

var utterances = ['tall', 'short', 'silence'];

var speaker = cache(function(state, tallThresh, shortThresh, cc) {
  return Infer({method: 'enumerate'}, function() {
    var utterance = uniformDraw(utterances);
    var ll = literalListener(utterance, tallThresh, shortThresh, cc);
    factor(alpha * ll.score(state));
    return utterance;
  });
});

var pragmaticListener = function(utterance, subjectParams) {
  return Infer({method: 'enumerate'}, function() {
    var cc = uniformDraw(['super', 'sub']);
    var tallThresh = uniformDraw(tallThresholds);
    var shortThresh = uniformDraw(shortThresholds);
    var state = sample(stateDist(cc === 'sub' ? 'sub' : 'super'));
    // entity is a basketball player: its state is drawn from subordinate distribution
    var entityState = sample(Categorical({vs: stateValues, ps: subProbs}));
    var sp = speaker(entityState, tallThresh, shortThresh, cc);
    observe(sp, utterance);
    return cc;
  });
};

var ANSWER = pragmaticListener('tall', subParams);