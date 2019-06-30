package com.wolfram.links.rlink.dataTypes.outTypes;

/**
 * 
 *  RLinkJ source code (c) 2011-2012, Wolfram Research, Inc. 
 *  
 *  
 *  
 *   This file is part of RLinkJ interface to JRI Java library.
 *
 *   RLinkJ is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU Lesser General Public License as 
 *   published by the Free Software Foundation, either version 2 of 
 *   the License, or (at your option) any later version.
 *
 *   RLinkJ is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU Lesser General Public 
 *   License along with RLinkJ. If not, see <http://www.gnu.org/licenses/>.
 *  
 * 
 * 
 *
 * @author Leonid Shifrin
 *
 */

import com.wolfram.links.rlink.exceptions.RLinkWrongTypeException;


public enum ROutTypes {
	INTEGER_VECTOR("integer"),
	DOUBLE_VECTOR("double"),
	COMPLEX_VECTOR("complex"),
	CHARACTER_VECTOR("character"),
	LOGICAL_VECTOR("logical"),
	LIST("list"),
	NULL("NULL"),
	ENVIRONMENT("environment"),
	CLOSURE("closure"),
	BUILTIN("builtin"),
	UNKNOWN("unknown")
	;
	
	private final String type;

	ROutTypes(String type) {
		this.type = type;
	}

	public static ROutTypes getTypeByClassName(String type) {
		for (ROutTypes tp : ROutTypes.values()) {
			if (tp.type.equals(type)) {
				return tp;
			}
		}
		return UNKNOWN;
	}
	
	public static IROutType getNewInstanceOfType(String type, String varnameOrCode){
		ROutTypes tp = getTypeByClassName(type);
		switch(tp){
		case INTEGER_VECTOR:return new RIntegerVectorOutType(varnameOrCode);
		case DOUBLE_VECTOR: return new RDoubleVectorOutType(varnameOrCode);
		case COMPLEX_VECTOR: return new RComplexVectorOutType(varnameOrCode);
		case CHARACTER_VECTOR: return new RCharacterVectorOutType(varnameOrCode);
		case LOGICAL_VECTOR: return new RLogicalVectorOutType(varnameOrCode);
		case LIST: return new RListOutType(varnameOrCode);
		case NULL: return new RNullOutType(varnameOrCode);
		case ENVIRONMENT: return new REnvironmentOutType(varnameOrCode);
		case CLOSURE: return new RClosureOutType(varnameOrCode);
		case BUILTIN: return new RBuiltinOutType(varnameOrCode);
		case UNKNOWN: return new RDeparsedCodeOutType(varnameOrCode);
		default:
			throw new RLinkWrongTypeException("The class of type "
					+type.toString()+
					" is not supported and can not be created");	
		}		
	}
	
	public String getStringType(){
		return this.type;
	}
}

