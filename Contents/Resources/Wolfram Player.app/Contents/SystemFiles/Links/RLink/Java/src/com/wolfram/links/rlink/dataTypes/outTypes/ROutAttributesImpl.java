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

import java.util.HashSet;
import java.util.Set;

import com.wolfram.links.rlink.RCodeGenerator;
import com.wolfram.links.rlink.RExecutor;
import com.wolfram.links.rlink.exceptions.RLinkErrorException;

public class ROutAttributesImpl extends RListOutType implements IROutAttributes {

	private final String bareVarNameOrCode;

	public ROutAttributesImpl(String expr) {
		super("attributes(" + expr + ")");
		this.bareVarNameOrCode = expr;
		this.getAttributes = false;
	}

	@Override
	public boolean transferAttributesFromR(RExecutor exec) {
		return this.rGet(exec);
	}

	@Override
	public Set<String> getAllAttributeNames(RExecutor exec) {
		RCharacterVectorOutType names = new RCharacterVectorOutType("names("
				+ this.getVariableNameOrCodeString() + ")");
		if (!names.rGet(exec)) {
			throw new RLinkErrorException("Error when executing R code: "
					+ this.getVariableNameOrCodeString());
		}
		Set<String> result = new HashSet<String>();
		for (String attName : names.getElements()) {
			result.add(attName);
		}
		return result;
	}

	@Override
	public boolean isAttribute(String attName, RExecutor exec) {
		// TODO We may want a more efficient way here
		return getAllAttributeNames(exec).contains(attName);
	}

	@Override
	public IROutType getAttribute(String attName, RExecutor exec) {
		String type = exec.evalGetString(RCodeGenerator.rGetAttributeType(
				this.bareVarNameOrCode, attName));
		IROutType result = ROutTypes.getNewInstanceOfType(type, RCodeGenerator
				.rGetAttributeCode(this.bareVarNameOrCode, attName));
		if (!result.rGet(exec)) {
			return null;
		}
		return result;
	}

	@Override
	public String toString() {
		return "ROutAttributesImpl [getList()=" + getList() + "]";
	}

}
