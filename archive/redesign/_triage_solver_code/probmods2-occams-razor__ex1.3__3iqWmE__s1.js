// Define the max number
var maxNumber = 20;
var allIntegers = _.range(1, maxNumber + 1);

// Define all concepts
// Rule-based: multiples of N for N in 1..11
var multiplesHypotheses = map(function(n) {
  return filter(function(x) { return x % n === 0; }, allIntegers);
}, _.range(1, 12));

// Rule-based: powers of N for N in 2..11 (N=1 would be just [1])
var powersHypotheses = map(function(n) {
  var powers = filter(function(x) {
    // check if x is a power of n
    var logVal = Math.log(x) / Math.log(n);
    return Math.abs(Math.round(logVal) - logVal) < 1e-9 && Math.round(logVal) >= 1;
  }, allIntegers);
  return powers;
}, _.range(2, 12));

// Also include powers of 1 (just [1])
var powersOf1 = [1];
var allPowersHypotheses = [powersOf1].concat(powersHypotheses);

// All evens
var allEvens = filter(function(x) { return x % 2 === 0; }, allIntegers);
// All odds
var allOdds = filter(function(x) { return x % 2 !== 0; }, allIntegers);

// All rule-based hypotheses
var ruleBasedHypotheses = multiplesHypotheses.concat(allPowersHypotheses).concat([allEvens, allOdds]);

// Interval concepts: all integers from a through b for every a < b in [1,20]
var intervalHypotheses = [];
var buildIntervals = function(a, acc) {
  if (a > maxNumber - 1) return acc;
  var newAcc = reduce(function(b, innerAcc) {
    if (b <= a) return innerAcc;
    var interval = filter(function(x) { return x >= a && x <= b; }, allIntegers);
    return innerAcc.concat([interval]);
  }, acc, allIntegers);
  return buildIntervals(a + 1, newAcc);
};
var intervalHypothesesList = buildIntervals(1, []);

// Prior: 50/50 mixture of rule-based and interval
// Within each class, equally likely
var numRuleBased = ruleBasedHypotheses.length;
var numInterval = intervalHypothesesList.length;

// observations
var observations = [3, 6, 9];

// For each hypothesis, compute the likelihood of observations
var computeLikelihood = function(concept) {
  var sz = concept.length;
  if (sz === 0) return 0;
  // Check all observations are in concept
  var allIn = reduce(function(obs, acc) {
    return acc && _.includes(concept, obs);
  }, true, observations);
  if (!allIn) return 0;
  return Math.pow(1.0 / sz, observations.length);
};

// Compute unnormalized posterior weights for all hypotheses
// Each rule-based hypothesis has prior weight: 0.5 / numRuleBased
// Each interval hypothesis has prior weight: 0.5 / numInterval

var ruleWeights = map(function(concept) {
  return (0.5 / numRuleBased) * computeLikelihood(concept);
}, ruleBasedHypotheses);

var intervalWeights = map(function(concept) {
  return (0.5 / numInterval) * computeLikelihood(concept);
}, intervalHypothesesList);

var allWeights = ruleWeights.concat(intervalWeights);
var allConcepts = ruleBasedHypotheses.concat(intervalHypothesesList);

var totalWeight = sum(allWeights);

// For each integer 1..20, compute posterior probability of membership
var membershipProbs = map(function(x) {
  var weightedMembership = sum(map(function(i) {
    var concept = allConcepts[i];
    var w = allWeights[i];
    return _.includes(concept, x) ? w : 0;
  }, _.range(allConcepts.length)));
  return weightedMembership / totalWeight;
}, allIntegers);

var ANSWER = membershipProbs;