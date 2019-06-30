
BeginPackage["CompileUtilities`Asserter`", {
	"CompileUtilities`Asserter`Assert`",
	"CompileUtilities`Asserter`Assume`",
	"CompileUtilities`Asserter`Expect`"
}]

(**
 * Borrows ideas from
 * http://www.dartdocs.org/documentation/matcher/0.11.4+1/index.html#matcher/matcher
 * http://google.github.io/truth/
 ************************************************************************************
 * Here is the gist from Wikipedia (http://en.wikipedia.org/wiki/Hamcrest)
 *  
 * "First generation" unit test frameworks provide an 'assert' statement, allowing
 * one to assert during a test that a particular condition must be true. If the
 * condition is false, the test fails. For example:
 * 
 *     assert(x == y);
 * 
 * But this syntax fails to produce a sufficiently good error message if 'x' and
 * 'y' are not equal. It would be better if the error message displayed the value
 * of 'x' and 'y'. To solve this problem, "second generation" unit test frameworks
 * provide a family of assertion statements, which produce better error messages.
 * For example,
 * 
 *     assert_equal(x, y);
 *     assert_not_equal(x, y);
 * 
 * But this leads to an explosion in the number of assertion macros, as the above
 * set is expanded to support comparisons different from simple equality. So
 * "third generation" unit test frameworks use a library such as Hamcrest to
 * support an 'assert_that' operator that can be combined with 'matcher' objects,
 * leading to syntax like this:
 * 
 *     assert_that(x, equal_to(y))
 *     assert_that(x, is_not(equal_to(y)))
 * 
 * The benefit is that you still get fluent error messages when the assertion
 * fails, but now you have greater extensibility. It is now possible to define
 * operations that take matchers as arguments and return them as results, leading
 * to a grammar that can generate a huge number of possible matcher expressions
 * from a small number of primitive matchers.
 * 
 * These higher-order matcher operations include logical connectives (and, or and
 * 	not), and operations for iterating over collections. This results in a
 * rich matcher language which allows complex assertions over collections to be
 * written in a declarative, rather than a procedural, programming style.
 * 
 *)


EndPackage[]
