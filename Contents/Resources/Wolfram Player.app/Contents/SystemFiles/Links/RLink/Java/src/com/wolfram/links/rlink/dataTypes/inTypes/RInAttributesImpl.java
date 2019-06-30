package com.wolfram.links.rlink.dataTypes.inTypes;

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

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import com.wolfram.links.rlink.RCodeGenerator;
import com.wolfram.links.rlink.RExecutor;

public class RInAttributesImpl implements IRInAttributes {

	private Map<String, IRInType> attributes = new HashMap<String, IRInType>();

	public RInAttributesImpl() {
		super();
	}

	public RInAttributesImpl(Map<String, IRInType> attributes) {
		super();
		this.attributes = attributes;
	}

	public RInAttributesImpl(int[] dims) {
		super();
		setAttribute("dim", new RIntegerVectorInType(dims, new int[0],
				new RInAttributesImpl()));
	}

	@Override
	public Set<String> getAllAttributeNames() {
		return attributes.keySet();
	}

	@Override
	public IRInType getAttribute(String attrName) {
		return attributes.get(attrName);
	}

	@Override
	public boolean isAttribute(String attrName) {
		return attributes.containsKey(attrName);
	}

	@Override
	public void setAttribute(String attrName, IRInType val) {
		attributes.put(attrName, val);
	}

	@Override
	public boolean transferAttributesToR(String rVar, RExecutor exec) {
		for (String attName : attributes.keySet()) {
			String code = RCodeGenerator.rGetAttributeCode(rVar, attName);
			if (!attributes.get(attName).rPut(code, exec)) {
				return false;
			}
		}
		return true;		
	}

}
