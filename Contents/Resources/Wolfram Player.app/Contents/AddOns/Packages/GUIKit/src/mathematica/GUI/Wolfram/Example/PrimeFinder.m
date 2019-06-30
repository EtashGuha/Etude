Widget["Panel", {

 WidgetGroup[{
 
  { WidgetFill[],
    Widget["Button", {
      "label" -> "Previous",
      BindEvent["action", 
        Script[
          findPrime[-1];
          ]
        ]}, Name -> "previousButton"],   
    WidgetSpace[5],
    Widget["Button", {
      "label" -> "Next",
      BindEvent["action", 
        Script[
          findPrime[1]
          ]
        ]}, Name -> "nextButton"],
     WidgetFill[]
    },

  Widget["TextField", {
     "text" -> "1",
     "horizontalAlignment" -> PropertyValue["RIGHT"],
     BindEvent["keyReleased", 
       Script[ 
         testIfPrime[];
         ]
      ]}, 
    Name -> "valueField",
    WidgetLayout -> {"Stretching" -> {Maximize, Maximize}}],
      
  Widget["Label", {
    "text" -> "Number is not prime."
      }, Name -> "primeIndicator"]
      
  }, WidgetLayout ->Column],
  
  BindEvent["endModal", 
    Script[
      Union[ ToExpression /@ primesVisited]
      ]
    ],
    
  Script[
  
    primesVisited = {};

    findPrime[direction_] := Module[{value},
      value = PropertyValue[{"valueField", "text"}];
      If[ !SyntaxQ[value], Return[]];
      value = Round[ ToExpression[value]];
      If[ !NumericQ[value], value = 0];
      value += direction;
      While[ !PrimeQ[value],
        value += direction ];
      SetPropertyValue[{"valueField", "text"}, ToString[value, InputForm] ];
      testIfPrime[];
      ];

    testIfPrime[] := Module[{value, isPrime = False},
      value = PropertyValue[{"valueField", "text"}];
      If[ SyntaxQ[value], 
        isPrime = PrimeQ[ToExpression[value]] ];
      If[ TrueQ[isPrime],
         primesVisited = Union[primesVisited, {ToString[value]}];
         SetWidgetReference["primesVisited", primesVisited ];
         ];
      SetPropertyValue[{"primeIndicator", "text"}, "Number is " <> If[ TrueQ[isPrime], "", "not "] <> "prime." ];
      ];
    
    ]
    
}]
