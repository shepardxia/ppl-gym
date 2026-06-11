var maxNumber = 20;
var allNums = _.range(1, maxNumber + 1);

// Build rule-based concepts
var multiples = map(function(n) {
  return filter(function(x) { return x % n === 0; }, allNums);
}, _.range(1, 12));

var powers = map(function(n) {
  return filter(function(x) {
    // is x a power of n? i.e. x = n^k for some k>=1 (and n=1 -> only 1)
    var isPow = function(v) {
      return v === x ? true : (v > x ? false : isPow(v * n));
    };
    return n === 1 ? (x === 1) : isPow(n);
  }, allNums);
}, _.range(1, 12));

var evens = [filter(function(x) { return x % 2 === 0; }, allNums)];
var odds = [filter(function(x) { return x % 2 === 1; }, allNums)];

var ruleConcepts = _.flatten([multiples, powers, evens, odds]);

// Build interval concepts: all a<b in [1,20]
var intervalPairs = _.flatten(map(function(a) {
  return map(function(b) {
    return [a, b];
  }, _.range(a + 1, maxNumber + 1));
}, _.range(1, maxNumber + 1)));

var intervalConcepts = map(function(pair) {
  return _.range(pair[0], pair[1] + 1);
}, intervalPairs);

var examples = [3, 6, 9];

var contains = function(concept, x) {
  return _.includes(concept, x);
};

// log likelihood under size principle
var logLik = function(concept) {
  var ok = reduce(function(e, acc) {
    return acc && contains(concept, e);
  }, true, examples);
  if (!ok) return -Infinity;
  return examples.length * (-Math.log(concept.length));
};

// log prior: 0.5 mixture, equal within class
var nRule = ruleConcepts.length;
var nInt = intervalConcepts.length;

var ruleLogPrior = Math.log(0.5) - Math.log(nRule);
var intLogPrior = Math.log(0.5) - Math.log(nInt);

var ruleScored = map(function(c) {
  return { concept: c, score: ruleLogPrior + logLik(c) };
}, ruleConcepts);

var intScored = map(function(c) {
  return { concept: c, score: intLogPrior + logLik(c) };
}, intervalConcepts);

var allScored = _.flatten([ruleScored, intScored]);

// normalize
var maxScore = reduce(function(s, acc) {
  return Math.max(s.score, acc);
}, -Infinity, allScored);

var weights = map(function(s) {
  return Math.exp(s.score - maxScore);
}, allScored);

var totalW = sum(weights);

var normWeights = map(function(w) { return w / totalW; }, weights);

// expected membership for each integer
var membership = map(function(x) {
  var contribs = mapIndexed(function(i, s) {
    return contains(s.concept, x) ? normWeights[i] : 0;
  }, allScored);
  return sum(contribs);
}, allNums);

var ANSWER = membership;