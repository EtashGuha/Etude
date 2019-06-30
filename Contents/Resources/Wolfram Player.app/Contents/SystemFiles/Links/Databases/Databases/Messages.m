Package["Databases`"]

Databases::interr = "Error of type `1` in internal function `2`, called with the arguments: `3`.";


DatabaseStore::nopk = "Table \"`1`\" doesn't have a primary key, DatabaseModelInstance won't work."

RelationalDatabase::nvldbackend = "Operation \"``\" is not supported for backend \"``\"."
RelationalDatabase::nvldtype = "Unsupported type `` for column \"``\" in table \"``\". No operation will be allowed in EntityFunction. Results might be unusable."

DatabaseReference::nvldconn  = "Invalid reference specification ``.";
DatabaseReference::nvldfile  = "No SQLite file found at location ``.";
DatabaseReference::nvldprop  = "Invalid property ``. Possible choices are: ``.";
DatabaseReference::nvldback  = "Invalid backend \"``\". Possible choices are: ``.";
DatabaseReference::emptyname = "No database name provided."
DatabaseReference::nvldvalue = "`` is not a valid value for property ``. Please insert a ``.";

DatabaseReference::conn = "Database connection was successful."
DatabaseReference::aconn = "Database was already connected."

DatabaseReference::disc = "Database disconnection was successful."
DatabaseReference::adisc = "Database was already disconnected."


RelationalDatabase::edited = "Manually editing RelationalDatabase is not supported."

(* Messages used on the python side *)
(* User faced messages *)

RelationalDatabase::typenvld     = "Invalid type name ``."
RelationalDatabase::typerequired = "An argument is required for ``."
RelationalDatabase::typeargcount = "`` is called with `` arguments. Max is ``."
RelationalDatabase::typeint      = "`` must be a valid integer not ``."
RelationalDatabase::typestring   = "`` must be a string not ``."
RelationalDatabase::typechoices  = "`` must be a list of values not ``."
RelationalDatabase::typeboolean  = "`` must be a True or False and not ``."
RelationalDatabase::nvldplatform = "This platform currently does not support SQL Database functionalities."
RelationalDatabase::resnodwnld   = "The required DatabasesResources paclet could \
not be downloaded. Make sure the computer is connected to the internet"
RelationalDatabase::intdisal     = "The required DatabasesResources paclet could \
not be downloaded, because internet access is switched off. Edit preferences to 
enable internet connectivity"
RelationalDatabase::interr       = "Unknown internal error"
RelationalDatabase::pcltinsterr  = "The required DatabasesResources paclet failed \
to install properly. Try to rerun the code, or get and install the paclet manually"

RelationalDatabase::inspfk = "Foreign key constraint on table \"``\" references table \"``\" which is not part of the inspected tables. It will not be present in the RelationalDatabase object.";
RelationalDatabase::insptb = "The database contains multiple tables named `` in schemas ``. Only the first one will be preserved."
RelationalDatabase::inspmtb = "Table \"``\" was not found."