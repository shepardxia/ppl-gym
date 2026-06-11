// Define the max number
var maxNumber = 20;
var allIntegers = _.range(1, maxNumber + 1);

// Build all hypotheses

// Rule-based: multiples of N for N in 1..11
var multiplesHypotheses = map(function(n) {
  return {
    type: 'multiples',
    n: n,
    extension: filter(function(x) { return x % n === 0; }, allIntegers)
  };
}, _.range(1, 12));

// Rule-based: powers of N for N in 1..11
var powersHypotheses = map(function(n) {
  var ext = filter(function(x) {
    // Check if x is a power of n
    if (n === 1) return x === 1;
    var val = 1;
    var isPower = false;
    while (val <= maxNumber) {
      if (val === x) { isPower = true; break; }
      val = val * n;
    }
    return isPower;
  }, allIntegers);
  return {
    type: 'powers',
    n: n,
    extension: ext
  };
}, _.range(1, 12));

// Rule-based: all evens
var evensHypothesis = {
  type: 'evens',
  extension: filter(function(x) { return x % 2 === 0; }, allIntegers)
};

// Rule-based: all odds
var oddsHypothesis = {
  type: 'odds',
  extension: filter(function(x) { return x % 2 !== 0; }, allIntegers)
};

var ruleBasedHypotheses = multiplesHypotheses.concat(powersHypotheses).concat([evensHypothesis, oddsHypothesis]);

// Interval concepts: all integers from a through b inclusive for every a < b in [1, 20]
var intervalHypotheses = _.flatten(map(function(a) {
  return map(function(b) {
    return {
      type: 'interval',
      a: a,
      b: b,
      extension: filter(function(x) { return x >= a && x <= b; }, allIntegers)
    };
  }, _.range(a + 1, maxNumber + 1));
}, allIntegers));

// Total counts
var numRuleBased = ruleBasedHypotheses.length;
var numInterval = intervalHypotheses.length;

// Prior: 50/50 mixture between rule-based and interval
// Each hypothesis equally likely within its class
// Prior weight of each rule-based: 0.5 / numRuleBased
// Prior weight of each interval: 0.5 / numInterval

// Observations
var examples = [3, 6, 9];

// Compute likelihood for a hypothesis given examples (size principle)
var likelihood = function(hyp) {
  var extSize = hyp.extension.length;
  if (extSize === 0) return 0;
  // Check if all examples are in extension
  var allCovered = all(function(e) { return _.includes(hyp.extension, e); }, examples);
  if (!allCovered) return 0;
  return Math.pow(1.0 / extSize, examples.length);
};

// Compute unnormalized posterior weights for all hypotheses
var ruleBasedWeights = map(function(hyp) {
  return (0.5 / numRuleBased) * likelihood(hyp);
}, ruleBasedHypotheses);

var intervalWeights = map(function(hyp) {
  return (0.5 / numInterval) * likelihood(hyp);
}, intervalHypotheses);

var allHypotheses = ruleBasedHypotheses.concat(intervalHypotheses);
var allWeights = ruleBasedWeights.concat(intervalWeights);

// Normalize
var totalWeight = sum(allWeights);

var normalizedWeights = map(function(w) { return w / totalWeight; }, allWeights);

// For each integer 1..20, compute posterior probability of membership
var membershipProbs = map(function(x) {
  var weightedMembership = sum(map(function(i) {
    var hyp = allHypotheses[i];
    var w = normalizedWeights[i];
    var isMember = _.includes(hyp.extension, x) ? 1 : 0;
    return w * isMember;
  }, _.range(allHypotheses.length)));
  return weightedMembership;
}, allIntegers);

var ANSWER = membershipProbs;