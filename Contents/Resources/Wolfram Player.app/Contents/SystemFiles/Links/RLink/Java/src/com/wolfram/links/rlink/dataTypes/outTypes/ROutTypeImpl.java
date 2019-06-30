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

import com.wolfram.links.rlink.RExecutor;

public abstract class ROutTypeImpl implements IROutType {
	protected IROutAttributes attributes = null;
	private final String expr;

	public ROutTypeImpl(String expr) {
		super();
		this.expr = expr;
	}

	@Override
	public String getVariableNameOrCodeString() {
		return expr;
	}

	public boolean rGet(RExecutor exec) {
		ROutTypes varType = ROutTypes.getTypeByClassName(exec
				.getRObjectType(this.expr));
		assert getType().equals(varType);
		if (!exec.hasAttributes(expr)) {
			return true;
		}
		attributes = new ROutAttributesImpl(this.expr);
		return attributes.transferAttributesFromR(exec);
	}

	@Override
	public IROutAttributes getAttributes() {
		return this.attributes;
	}
}
