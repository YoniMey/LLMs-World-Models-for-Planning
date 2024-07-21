ACTION_PARAMS_PROMPT_BASE = """
You are defining the objects, preconditions and effects (represented in PDDL format) of an AI agent's actions. Information about the AI agent will be provided in the domain description.
you are going to do it step by step.

At this step you need to define the objects on which the action will operate.

#######################################

Here are two examples from the classical BlocksWorld domain for demonstrating the output format:

Domain information: BlocksWorld is a planning domain in artificial intelligence. The AI agent here is a mechanical robot arm that can pick and place the blocks. Only one block may be moved at a time: it may either be placed on the table or placed atop another block. Because of this, any blocks that are, at a given time, under another block cannot be moved. There is only one type of object in this domain, and that is the block.

Example 1
Action: This action enables the robot to put a block onto the table. For example, the robot puts block_1 onto the table.

Parameters:
1. ?x - block: the block to put down

#######################################

Example 2
Action: This action enables the robot to pick up a block on the table.

Parameters:
1. ?x - block: the block to pick up

#######################################

Here is the task:
Domain information: 
"""


ACTION_PRECONDITIONS_PROMPT_BASE = """

you're now at the second step. you will provide the objects for the action and now you need to define the preconditions (represented in PDDL format) of the AI agent's actions. The preconditions is a list of predicates that represent what must be true to operate the action.
Note that individual conditions in preconditions should be listed separately. For example, "object_1 is washed and heated" should be considered as two separate conditions "object_1 is washed" and "object_1 is heated". Also, in PDDL, two predicates cannot have the same name even if they have different parameters. Each predicate in PDDL must have a unique name, and its parameters must be explicitly defined in the predicate definition. It is recommended to define predicate names in an intuitive and readable way.

#######################################

Here are two examples from the classical BlocksWorld domain for demonstrating the output format:

Domain information: BlocksWorld is a planning domain in artificial intelligence. The AI agent here is a mechanical robot arm that can pick and place the blocks. Only one block may be moved at a time: it may either be placed on the table or placed atop another block. Because of this, any blocks that are, at a given time, under another block cannot be moved. There is only one type of object in this domain, and that is the block.

#######################################

Example 1
Action: This action enables the robot to put a block onto the table. For example, the robot puts block_1 onto the table.

the parameters for the action are:
1. ?x - block: the block to put down

you can still add or remove parameters if you think that it is needed.

You can create and define new predicates, but you may also reuse the following predicates:
(robot-holding ?x)

Parameters:
1. ?x - block: the block to put down

Preconditions:

(and
    (robot-holding ?x)
)

New Predicates:
No newly defined predicate


Example 2
Action: This action enables the robot to pick up a block on the table.

the parameters for the action are:
1. ?x - block: the block to pick up

you can still add or remove parameters if you think that it is needed.

You can create and define new predicates, but you may also reuse the following predicates:
1. (robot-holding ?x - block): true if the robot arm is holding the block ?x

Parameters:
1. ?x - block: the block to pick up

Preconditions:

(and
    (block-clear ?x)
    (block-on-table ?x)
    (robot-hand-empty)
)
\


New Predicates:
(block-clear ?x)
(block-on-table ?x)
(robot-hand-empty)

Here is the task:

"""

ACTION_EFFECTS_PROMPT_BASE = """
you're now at the third and last step. you will provide the objects and the preconditions  for the action and now you need to define the effects (represented in PDDL format) of the AI agent's actions. The effects are a list of predicates that represent what changes in the state of the world after the action operated.
Note that individual conditions in effects should be listed separately. For example, "object_1 is washed and heated" should be considered as two separate conditions "object_1 is washed" and "object_1 is heated". Also, in PDDL, two predicates cannot have the same name even if they have different parameters. Each predicate in PDDL must have a unique name, and its parameters must be explicitly defined in the predicate definition. It is recommended to define predicate names in an intuitive and readable way.

#######################################

Here are two examples from the classical BlocksWorld domain for demonstrating the output format:

Domain information: BlocksWorld is a planning domain in artificial intelligence. The AI agent here is a mechanical robot arm that can pick and place the blocks. Only one block may be moved at a time: it may either be placed on the table or placed atop another block. Because of this, any blocks that are, at a given time, under another block cannot be moved. There is only one type of object in this domain, and that is the block.

#######################################

Example 1:
Action: This action enables the robot to put a block onto the table. For example, the robot puts block_1 onto the table.

the parameters for this action are:
1. ?x - block: the block to put down

you can still add or remove parameters if you think that it is needed.

the preconditions for this action are:
(and
    (robot-holding ?x)
)

you can still add or remove preconditions if you think that it is needed.

You can create and define new predicates, but you may also reuse the following predicates:
No predicate has been defined yet

Parameters:
1. ?x - block: the block to put down

Preconditions:

(and
    (robot-holding ?x)
)

Effects:

(and
    (not (robot-holding ?x))
    (block-clear ?x)
    (robot-hand-empty)
    (block-on-table ?x)
)


New Predicates:
1. (robot-holding ?x - block): true if the robot arm is holding the block ?x
2. (block-clear ?x - block): true if the block ?x is not under any another block
3. (robot-hand-empty): true if the robot arm is not holding any block
4. (block-on-table ?x - block): true if the block ?x is placed on the table

#######################################

Example 2
Action: This action enables the robot to pick up a block on the table.

the parameters for this action are:
1. ?x - block: the block to pick up

you can still add or remove parameters if you think that it is needed.

the preconditions for this action are:
(and
    (block-clear ?x)
    (block-on-table ?x)
    (robot-hand-empty)
)


you can still add or remove preconditions if you think that it is needed.

You can create and define new predicates, but you may also reuse the following predicates:
1. (robot-holding ?x - block): true if the robot arm is holding the block ?x
2. (block-clear ?x - block): true if the block ?x is not under any another block
3. (robot-hand-empty): true if the robot arm is not holding any block
4. (block-on-table ?x - block): true if the block ?x is placed on the table

Parameters:
1. ?x - block: the block to pick up

Preconditions:

(and
    (block-clear ?x)
    (block-on-table ?x)
    (robot-hand-empty)
)

Effects:

(and
    (not (block-on-table ?x))
    (not (block-clear ?x))
    (not (robot-hand-empty))
    (robot-holding ?x)
)

New Predicates:
No newly defined predicate

Here is the task:

"""