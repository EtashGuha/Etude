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

import java.util.List;
import java.util.ListIterator;

import com.wolfram.links.rlink.RCodeGenerator;
import com.wolfram.links.rlink.RExecutor;

public class RListInType extends RInTypeImpl {
	private final List<IRInType> list;

	public RListInType(List<IRInType> list, IRInAttributes attributes) {
		super(attributes);
		this.list = list;
	}

	@Override
	public boolean rPut(String rVar, RExecutor exec) {
		exec.eval(rVar + "<- list()");
		exec.eval("attributes(" + rVar + ") <- NULL");
		ListIterator<IRInType> iter = list.listIterator();
		int i = 1;
		while (iter.hasNext()) {
			if (!iter.next().rPut(RCodeGenerator.listIndex(rVar, i), exec)) {
				return false;
			}
			i++;
		}
		return super.rPut(rVar, exec);
	}

}
