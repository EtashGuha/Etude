package com.wolfram.databaselink.examples;

import java.util.EventObject;

public class CommandAddedEvent extends EventObject
{
  private Command command;

  public CommandAddedEvent(Object source, Command command)
  {
    super(source);
    this.command = command;
  }

  public Command getCommand()
  {
    return command;
  }
}


