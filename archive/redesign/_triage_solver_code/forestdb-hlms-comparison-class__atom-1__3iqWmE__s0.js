var binParam = 3;
var alpha = 5;

// State values: 18 evenly spaced points from -3 to 3 (exclusive) in steps of 1/3
var stateValues = map(function(i) { return -3 + (1/3) * i; }, _.range(18));

// Gaussian PDF
var gaussianPDF = function(x, mu, sigma) {
  return Math.exp(-0.5 * Math.pow((x - mu) / sigma, 2)) / (sigma * Math.sqrt(2 * Math.PI));
};

// State probabilities for superordinate (mu=0, sigma=1)
var superordinateProbs = function() {
  var probs = map(function(s) { return gaussianPDF(s, 0, 1); }, stateValues);
  var total = sum(probs);
  return map(function(p) { return p / total; }, probs);
};

// State probabilities for subordinate/basketball (mu=1, sigma=0.5)
var subordinateProbs = function() {
  var probs = map(function(s) { return gaussianPDF(s, 1, 0.5); }, stateValues);
  var total = sum(probs);
  return map(function(p) { return p / total; }, probs);
};

var superProbsArr = superordinateProbs();
var subProbsArr = subordinateProbs();

// Tall thresholds: each state value minus 1/(2*binParam)
var tallThresholds = map(function(s) { return s - 1/(2*binParam); }, stateValues);
// Short thresholds: each state value plus 1/(2*binParam)
var shortThresholds = map(function(s) { return s + 1/(2*binParam); }, stateValues);

// Literal listener: given utterance and comparison class, infers state
var literalListener = mem(function(utterance, compClass) {
  return Infer({method: 'enumerate'}, function() {
    var stateIndex = randomInteger(18);
    var state = stateValues[stateIndex];
    var probs = compClass === 'superordinate' ? superProbsArr : subProbsArr;
    factor(Math.log(probs[stateIndex]));
    
    // Condition on utterance truth
    if (utterance === 'tall') {
      var tallThreshIndex = randomInteger(18);
      var tallThresh = tallThresholds[tallThreshIndex];
      condition(state > tallThresh);
    } else if (utterance === 'short') {
      var shortThreshIndex = randomInteger(18);
      var shortThresh = shortThresholds[shortThreshIndex];
      condition(state < shortThresh);
    }
    // silence: always true
    
    return state;
  });
});

// Speaker: given state, tall threshold, short threshold, comparison class
var speaker = mem(function(state, tallThresh, shortThresh, compClass) {
  return Infer({method: 'enumerate'}, function() {
    var utterance = categorical({vs: ['tall', 'short', 'silence'], ps: [1/3, 1/3, 1/3]});
    
    // Check utterance truth
    var isTall = state > tallThresh;
    var isShort = state < shortThresh;
    var isTrue = (utterance === 'tall') ? isTall : 
                 (utterance === 'short') ? isShort : true;
    
    condition(isTrue);
    
    // Factor by literal listener's log probability of state
    var llDist = literalListener(utterance, compClass);
    factor(alpha * llDist.score(state));
    
    return utterance;
  });
});

// Pragmatic listener hears 'tall' about a basketball player
// Infers: state, tallThresh, shortThresh, compClass
// The entity is a basketball player, so we use subordinate distribution for the entity's state prior
var pragmaticListener = Infer({method: 'enumerate'}, function() {
  // Draw state from basketball player (subordinate) distribution
  var stateIndex = randomInteger(18);
  var state = stateValues[stateIndex];
  factor(Math.log(subProbsArr[stateIndex]));
  
  // Draw thresholds
  var tallThreshIndex = randomInteger(18);
  var tallThresh = tallThresholds[tallThreshIndex];
  
  var shortThreshIndex = randomInteger(18);
  var shortThresh = shortThresholds[shortThreshIndex];
  
  // Draw comparison class uniformly
  var compClass = flip(0.5) ? 'superordinate' : 'subordinate';
  
  // Observe utterance 'tall'
  var speakerDist = speaker(state, tallThresh, shortThresh, compClass);
  observe(speakerDist, 'tall');
  
  return compClass;
});

var ANSWER = pragmaticListener;