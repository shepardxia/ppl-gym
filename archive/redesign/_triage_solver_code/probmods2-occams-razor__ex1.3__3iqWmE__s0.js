// Define the max number
var maxNumber = 20;
var integers = _.range(1, maxNumber + 1);

// Helper: extension of a concept
var extension = function(concept) {
  if (concept.type === 'multiples') {
    return filter(function(x) { return x % concept.n === 0; }, integers);
  } else if (concept.type === 'powers') {
    // powers of n: n^1, n^2, ... within range
    var pows = [];
    var buildPows = function(val) {
      return val > maxNumber ? [] : [val].concat(buildPows(val * concept.n));
    };
    return concept.n === 1 ? [1] : buildPows(concept.n);
  } else if (concept.type === 'evens') {
    return filter(function(x) { return x % 2 === 0; }, integers);
  } else if (concept.type === 'odds') {
    return filter(function(x) { return x % 2 !== 0; }, integers);
  } else if (concept.type === 'interval') {
    return filter(function(x) { return x >= concept.a && x <= concept.b; }, integers);
  }
  return [];
};

// Build list of rule-based concepts
var ruleBasedConcepts = (function() {
  var multiples = map(function(n) { return {type: 'multiples', n: n}; }, _.range(1, 12));
  var powers = map(function(n) { return {type: 'powers', n: n}; }, _.range(2, 12));
  var evens = [{type: 'evens'}];
  var odds = [{type: 'odds'}];
  return multiples.concat(powers).concat(evens).concat(odds);
})();

// Build list of interval concepts
var intervalConcepts = (function() {
  var pairs = [];
  var buildIntervals = function(a, b) {
    if (a > maxNumber - 1) return [];
    if (b > maxNumber) return buildIntervals(a + 1, a + 2);
    return [{type: 'interval', a: a, b: b}].concat(buildIntervals(a, b + 1));
  };
  return buildIntervals(1, 2);
})();

var numRuleBased = ruleBasedConcepts.length;
var numInterval = intervalConcepts.length;

// All concepts
var allConcepts = ruleBasedConcepts.concat(intervalConcepts);

// Examples
var examples = [3, 6, 9];

// For each concept, compute log likelihood of examples
var logLikelihood = function(concept) {
  var ext = extension(concept);
  var n = ext.length;
  if (n === 0) return -Infinity;
  // Check all examples are in extension
  var allIn = all(function(x) { return _.includes(ext, x); }, examples);
  if (!allIn) return -Infinity;
  return -examples.length * Math.log(n);
};

// Prior: 50/50 between rule-based and interval, uniform within each class
var logPriorRuleBased = function(i) {
  return Math.log(0.5) - Math.log(numRuleBased);
};
var logPriorInterval = function(i) {
  return Math.log(0.5) - Math.log(numInterval);
};

// Compute unnormalized log posterior for each concept
var logPosteriors = map(function(concept) {
  var idx = allConcepts.indexOf(concept);
  var logPrior = idx < numRuleBased ? logPriorRuleBased(idx) : logPriorInterval(idx - numRuleBased);
  var ll = logLikelihood(concept);
  return logPrior + ll;
}, allConcepts);

// Normalize
var maxLogP = reduce(function(a, b) { return Math.max(a, b); }, -Infinity, logPosteriors);
var unnormalized = map(function(lp) { return Math.exp(lp - maxLogP); }, logPosteriors);
var Z = sum(unnormalized);
var posteriors = map(function(u) { return u / Z; }, unnormalized);

// For each test integer, compute expected membership
var membershipProbs = map(function(x) {
  var weightedMembership = sum(map(function(i) {
    var concept = allConcepts[i];
    var ext = extension(concept);
    var isMember = _.includes(ext, x) ? 1 : 0;
    return posteriors[i] * isMember;
  }, _.range(allConcepts.length)));
  return weightedMembership;
}, integers);

var ANSWER = membershipProbs;