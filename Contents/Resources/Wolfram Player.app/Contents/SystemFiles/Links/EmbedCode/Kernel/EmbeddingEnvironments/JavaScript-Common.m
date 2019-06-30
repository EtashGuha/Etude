(* This "Common" file is intended to be shared by all JavaScript language implementations (e.g., "JavaScript", "JavaScript-Nodejs", etc.)
   
   Pick a unique context. Other files that call functions defined here will use their fully-qualified context names.
*)

Begin["EmbedCode`JavaScript`Common`"]

(* Mappings from types recognized by the Interpreter[] function (these are the native WL-side types for APIFunction calls)
  to JavaScript native types.
*)
interpreterTypeToJavaScriptType["Integer" | "Integer8" | "Integer16" | "Integer32" | Integer | "Integer64" | "Number"] = "Number";
interpreterTypeToJavaScriptType[_] = "String";


(* Convert JS types to WL expressions. This is intended for use when the APIFunction arg slot is typed as Expression. *)
JStoWLFunction =
"var _JStoWL = function(_obj) {
        if (_obj instanceof Array) {
            var _s = \"{\";
            for (var _i = 0; _i < _obj.length; _i++) {
                 _s += _JStoWL(_obj[_i]);
                if (_i < _obj.length - 1)
                    _s += \",\";
            }
            _s += \"}\";
            return _s;
        } else if (typeof(_obj) === \"number\") {
            return _obj;
        } else if (typeof(_obj) === \"boolean\") {
            return _obj ? \"True\" : \"False\";
        } else {
            return \"\\\"\" + _obj + \"\\\"\";
        }
    }
"

End[]