// Define the max number
var maxNumber = 20;
var allIntegers = _.range(1, maxNumber + 1);

// Define rule-based concepts
var ruleConcepts = (function() {
  var concepts = [];
  
  // Multiples of N for N in 1..11
  map(function(n) {
    var ext = filter(function(x) { return x % n === 0; }, allIntegers);
    if (ext.length > 0) concepts.push({name: 'multiples_' + n, extension: ext});
  }, _.range(1, 12));
  
  // Powers of N for N in 2..11 (N^1, N^2, ... within range)
  // Also N=1 would be just [1], let's include 1..11
  map(function(n) {
    var ext = [];
    var p = n;
    while (p <= maxNumber) {
      ext.push(p);
      p = p * n;
    }
    if (ext.length > 0) concepts.push({name: 'powers_' + n, extension: ext});
  }, _.range(1, 12));
  
  // All evens
  var evens = filter(function(x) { return x % 2 === 0; }, allIntegers);
  concepts.push({name: 'evens', extension: evens});
  
  // All odds
  var odds = filter(function(x) { return x % 2 !== 0; }, allIntegers);
  concepts.push({name: 'odds', extension: odds});
  
  return concepts;
})();

// Define interval concepts: all integers from a to b inclusive for every a < b in [1,20]
var intervalConcepts = (function() {
  var concepts = [];
  map(function(a) {
    map(function(b) {
      if (a < b) {
        var ext = _.range(a, b + 1);
        concepts.push({name: 'interval_' + a + '_' + b, extension: ext});
      }
    }, allIntegers);
  }, allIntegers);
  return concepts;
})();

var numRuleConcepts = ruleConcepts.length;
var numIntervalConcepts = intervalConcepts.length;

var observations = [3, 6, 9];

// Compute log likelihood for a concept given observations (size principle)
var logLikelihood = function(concept, obs) {
  var extSize = concept.extension.length;
  // Check if all observations are in the extension
  var allInExt = all(function(o) {
    return _.includes(concept.extension, o);
  }, obs);
  
  if (!allInExt) return -Infinity;
  return -obs.length * Math.log(extSize);
};

// Compute unnormalized log posterior for each concept
// Prior: 50% rule-based, 50% interval
// Within each class, uniform over concepts

var ruleLogPrior = Math.log(0.5) - Math.log(numRuleConcepts);
var intervalLogPrior = Math.log(0.5) - Math.log(numIntervalConcepts);

var ruleLogPosteriors = map(function(c) {
  return ruleLogPrior + logLikelihood(c, observations);
}, ruleConcepts);

var intervalLogPosteriors = map(function(c) {
  return intervalLogPrior + logLikelihood(c, observations);
}, intervalConcepts);

var allLogPosteriors = ruleLogPosteriors.concat(intervalLogPosteriors);
var allConcepts = ruleConcepts.concat(intervalConcepts);

// Normalize
var maxLogPost = Math.max.apply(null, filter(function(x) { return x !== -Infinity; }, allLogPosteriors));

var unnormalized = map(function(lp) {
  return lp === -Infinity ? 0 : Math.exp(lp - maxLogPost);
}, allLogPosteriors);

var totalUnnorm = sum(unnormalized);

var posteriorWeights = map(function(u) { return u / totalUnnorm; }, unnormalized);

// For each integer 1..20, compute posterior expected membership
var membershipProbs = map(function(x) {
  var weightedMembership = sum(map(function(i) {
    var inExt = _.includes(allConcepts[i].extension, x) ? 1 : 0;
    return posteriorWeights[i] * inExt;
  }, _.range(0, allConcepts.length)));
  return weightedMembership;
}, allIntegers);

var ANSWER = membershipProbs;