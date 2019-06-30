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

import java.util.ArrayList;
import java.util.List;

import com.wolfram.links.rlink.RCodeGenerator;
import com.wolfram.links.rlink.RExecutor;
import com.wolfram.links.rlink.exceptions.RLinkWrongTypeException;

public class RListOutType extends ROutTypeImpl {

	public RListOutType(String expr) {
		super(expr);
	}

	private List<IROutType> list = new ArrayList<IROutType>();
	protected boolean getAttributes = true;

	@Override
	public boolean rGet(RExecutor exec) {
		int length = exec.getLength(this.getVariableNameOrCodeString());
		for (int i = 1; i <= length; i++) {
			String indexedElement = RCodeGenerator.listIndex(this
					.getVariableNameOrCodeString(), i);
			String type = exec.evalGetString(RCodeGenerator
					.rGetType(indexedElement));
			if (type == null) {
				return false;
			}
			IROutType element = null;
			try {
				element = ROutTypes.getNewInstanceOfType(type, indexedElement);
			} catch (RLinkWrongTypeException e) {
				// TODO Handle it differently?
				return false;
			}
			if (!element.rGet(exec)) {
				return false;
			}
			this.list.add(element);
		}
		return (!getAttributes) ? true : super.rGet(exec);
	}

	public List<IROutType> getList() {
		return list;
	}

	@Override
	public String toString() {
		return "RListOutType [list=" + list + ", attributes=" + attributes
				+ "]";
	}

	@Override
	public ROutTypes getType() {
		return ROutTypes.LIST;
	}

}
