import configparser
import datetime
import logging.config
import os
import random
from typing import Dict, Any, Union, List

from telegram import Update
from telegram.ext import (
    BasePersistence,
    Updater,
    CommandHandler,
    MessageHandler,
    Filters,
    ConversationHandler,
    CallbackContext,
    PicklePersistence
)

from classifier_abstract import ClassifierAbstract
from models.emotions_detection import SingleEmotionClassifier
from models.emotions_detection_custom import MultipleEmotionsClassifier
from models.generator_utils import answer_is_full_init, generate_questions_init, get_summary_init

logging.config.fileConfig('logging.conf')
logger = logging.getLogger(__name__)


def single_cl_output(cl_result: str) -> str:
    return cl_result


def multiple_cl_output(cl_result: Dict) -> str:
    return '\n'.join([f"{k} {v}" for k, v in cl_result.items()])


class Bot:
    STANDARD, GENERATED = range(2)

    def __init__(self,
                 token: str,
                 classifiers: Dict[str, ClassifierAbstract],
                 standard_questions: List[str],
                 def_cls: str = None,
                 st_q: int = 2,
                 gen_q: int = 3,
                 blocks: int = 2,
                 persistence: BasePersistence = None,
                 timeout: Union[float, datetime.timedelta] = None):
        self.__token = token
        if st_q <= 0 or gen_q <= 0 or blocks <= 0:
            raise ValueError("st_q, gen_q, blocks must be greater than 0")
        self.__question_amount = {self.STANDARD: st_q, self.GENERATED: gen_q}
        self.__blocks = blocks
        self.__standard_questions = standard_questions
        self.__generators = {'answer_is_full': answer_is_full_init(),
                             'generate_questions': generate_questions_init(),
                             'get_summary': get_summary_init()}
        if not classifiers:
            raise ValueError("classifiers dictionary shouldn't be empty")
        self.__classifiers = classifiers
        self.__default_classifier = def_cls if def_cls and def_cls in classifiers else next(iter(classifiers))
        self.__persistence = persistence
        self.__timeout = timeout
        logger.info("Bot class is set with classifiers: %s", list(self.__classifiers.keys()))

    def run(self):
        updater = Updater(self.__token, persistence=self.__persistence)
        dispatcher = updater.dispatcher
        conv_handler = ConversationHandler(
            name='BotSession',
            entry_points=[CommandHandler(['start_session', 'restart_session'], self.start_session)],
            states={
                self.STANDARD: [MessageHandler(Filters.text & ~Filters.command, self.answer_standard)],
                self.GENERATED: [MessageHandler(Filters.text & ~Filters.command, self.answer_generated)]
            },
            fallbacks=[CommandHandler('end_session', self.end_session)],
            allow_reentry=True,
            conversation_timeout=self.__timeout,
            persistent=bool(self.__persistence),
        )

        dispatcher.add_handler(conv_handler)
        dispatcher.add_handler(CommandHandler("help", self.help_command))
        dispatcher.add_handler(CommandHandler("start", self.start_command))
        dispatcher.add_handler(CommandHandler("last_session_stats", self.last_session_stats))
        dispatcher.add_handler(CommandHandler("sessions_stats", self.sessions_stats))
        dispatcher.add_handler(CommandHandler("set_classifier", self.set_classifier))
        dispatcher.add_handler(CommandHandler("get_classifiers", self.get_classifiers))

        updater.start_polling()
        updater.idle()

    def start_session(self, update: Update, context: CallbackContext) -> int:
        context.chat_data.setdefault('CLASSIFIER', self.__default_classifier)
        context.chat_data.setdefault('QUESTION_AMOUNT', self.__question_amount)
        context.chat_data.setdefault('BLOCKS', self.__blocks)
        context.user_data.setdefault('SESSIONS', [])

        logger.info("Started session by user: %s - %s in chat: %s",
                    update.effective_user.id, update.effective_user.username, update.effective_chat.id)
        context.user_data['block'] = 1
        context.user_data['messages'] = []
        context.user_data['message_buffer'] = []
        st_questions = self.__standard_questions
        random.shuffle(st_questions)
        context.user_data['standard_questions'] = st_questions
        context.user_data['generated_questions'] = []
        context.user_data['questions_amounts'] = dict.fromkeys(context.chat_data['QUESTION_AMOUNT'], 0)

        context.user_data['start_time'] = datetime.datetime.now()

        total_questions = sum(context.chat_data['QUESTION_AMOUNT'].values()) * context.chat_data['BLOCKS']
        update.message.reply_text("Hello,today we are gonna have some small talk about your feelings.\n"
                                  f"I'll ask you about {total_questions} questions to understand your emotional state.\n"
                                  "Be honest and give me as much detailed answer as you can.")
        question = context.user_data['standard_questions'].pop()
        update.message.reply_text(question)
        context.user_data['questions_amounts'][self.STANDARD] += 1

        logger.info("u: %d-%s c: %s | b: %d t: %s n: %d QUESTION: %s",
                    update.effective_user.id, update.effective_user.username, update.effective_chat.id,
                    context.user_data['block'], 'STANDARD', context.user_data['questions_amounts'][self.STANDARD],
                    question)

        return self.STANDARD

    def answer_standard(self, update: Update, context: CallbackContext) -> int:
        answer = update.message.text
        context.user_data['messages'].append(answer)
        context.user_data['message_buffer'].append(answer)
        answer_is_full = self.__generators['answer_is_full'](answer)
        more_questions = context.user_data['questions_amounts'][self.STANDARD] < context.chat_data['QUESTION_AMOUNT'][
            self.STANDARD]
        if not answer_is_full:
            question = "Could you kindly include some more details to your answer please?"
            q_type = 'CLARIFYING'
            state = self.STANDARD
        else:
            if more_questions:
                question = context.user_data['standard_questions'].pop()
                context.user_data['questions_amounts'][self.STANDARD] += 1
                q_type = 'STANDARD'
                state = self.STANDARD
            else:
                summary = self.__generators['get_summary']('. '.join(context.user_data['message_buffer']))
                question = ("Let's talk a bit more about what you've mentioned before.\n"
                            f"You said, '{summary}'. What else could you say about this?")
                q_type = 'ADDITIONAL'
                state = self.GENERATED

        logger.info("u: %d-%s c: %s | b: %d t: %s n: %d QUESTION: %s",
                    update.effective_user.id, update.effective_user.username, update.effective_chat.id,
                    context.user_data['block'], q_type, context.user_data['questions_amounts'][self.STANDARD],
                    question)
        update.message.reply_text(question)
        return state

    def answer_generated(self, update: Update, context: CallbackContext) -> int:
        answer = update.message.text
        context.user_data['messages'].append(answer)
        context.user_data['message_buffer'].append(answer)
        first = context.user_data['questions_amounts'][self.GENERATED] == 0
        answer_is_full = self.__generators['answer_is_full'](answer)
        more_questions = len(context.user_data['generated_questions']) > 0
        if first:
            context.user_data['generated_questions'] = self.__generators['generate_questions'](
                '. '.join(context.user_data['message_buffer']), context.chat_data['QUESTION_AMOUNT'][self.GENERATED])
        if not answer_is_full:
            question = "Could you kindly include some more details to your answer please?"
            q_type = 'CLARIFYING'
            state = self.GENERATED
        else:
            if more_questions:
                question = context.user_data['generated_questions'].pop()
                context.user_data['questions_amounts'][self.GENERATED] += 1
                q_type = 'GENERATED'
                state = self.GENERATED
            elif context.user_data['block'] < context.chat_data['BLOCKS']:
                context.user_data['message_buffer'] = []
                context.user_data['block'] += 1
                context.user_data['questions_amounts'] = dict.fromkeys(context.user_data['questions_amounts'], 0)
                question = context.user_data['standard_questions'].pop()
                context.user_data['questions_amounts'][self.STANDARD] += 1
                q_type = 'STANDARD'
                state = self.STANDARD
            else:
                classifier_alias = context.chat_data.get('CLASSIFIER', self.__default_classifier)
                classifier = self.__classifiers[classifier_alias]
                result = classifier.classify('. '.join(context.user_data['messages']))
                session = {'start_time': context.user_data['start_time'],
                           'end_time': datetime.datetime.now(),
                           'classifier': classifier_alias,
                           'result': result}
                context.user_data.setdefault('SESSIONS', []).append(session)
                output_f = classifier.output if hasattr(classifier, 'output') else None
                processed_result = output_f(result) if callable(output_f) else result
                update.message.reply_text(
                    ("Thank you so much! Now let's talk about your results:\n" + processed_result +
                     "\nThanks for using this bot!"))
                logger.info("u: %d-%s c: %s | CONVERSATION ENDED",
                            update.effective_user.id, update.effective_user.username, update.effective_chat.id)
                return ConversationHandler.END
        logger.info("u: %d-%s c: %s | b: %d t: %s n: %d QUESTION: %s",
                    update.effective_user.id, update.effective_user.username, update.effective_chat.id,
                    context.user_data['block'], q_type, context.user_data['questions_amounts'][self.STANDARD],
                    question)
        update.message.reply_text(question)
        return state

    def end_session(self, update: Update, context: CallbackContext) -> int:
        update.message.reply_text('You have ended this session, but you still can start a new one '
                                  'anytime you want with /start_session')
        logger.info("u: %d-%s c: %s | CONVERSATION TERMINATED",
                    update.effective_user.id, update.effective_user.username, update.effective_chat.id)
        return ConversationHandler.END

    def help_command(self, update: Update, context: CallbackContext) -> None:
        logger.info("u: %d-%s c: %s | HELP COMMAND CALLED",
                    update.effective_user.id, update.effective_user.username, update.effective_chat.id)
        update.message.reply_text('/start_session will start a conversation during which you have to answer questions '
                                  'one at a time with single message')

    def start_command(self, update: Update, context: CallbackContext) -> None:
        logger.info("u: %d-%s c: %s | START COMMAND CALLED",
                    update.effective_user.id, update.effective_user.username, update.effective_chat.id)
        update.message.reply_text("This bot was created to evaluate your emotional state.\nPrint /start_session "
                                  "to start the conversation, after that you will be asked to answer several "
                                  "questions. Depending on your responses, the bot can send some clarifying "
                                  "questions. At the end of the session, you'll receive emotions found in your "
                                  "responses.\nPlease be honest and clear and give us a detailed comment for "
                                  "each question.")

    def last_session_stats(self, update: Update, context: CallbackContext) -> None:
        sessions = context.user_data.setdefault('SESSIONS', [])
        if len(sessions) == 0:
            update.message.reply_text("Sorry, you don't have any sessions yet. Start one with /start_session command")
        else:
            s = sessions[-1]
            s_result = s.get('result', '')
            if 'classifier' in s:
                classifier = self.__classifiers[s.get('classifier', '')]
                output_f = classifier.output if hasattr(classifier, 'output') else None
                if callable(output_f):
                    s_result = output_f(s_result)
            update.message.reply_text(
                f"You're last session ended on {s.get('end_time', '')} using '{s.get('classifier', 'unknown')}' "
                f"classifier with result:\n{s_result}")

    def sessions_stats(self, update: Update, context: CallbackContext) -> None:
        sessions = context.user_data.setdefault('SESSIONS', [])
        if len(sessions) == 0:
            update.message.reply_text("Sorry, you don't have any sessions yet. Start one with /start_session command")
        else:
            stats_str = '\n'.join([(f"[{s.get('start_time', '')} - {s.get('end_time', '')}]: "
                                    f"{s.get('classifier', '')} |\t{s.get('result', '')}") for s in sessions])
            update.message.reply_text("Sessions stats:\n" + stats_str)

    def set_classifier(self, update: Update, context: CallbackContext) -> None:
        new_value = next(iter(context.args or []), None)
        if new_value is None or new_value not in self.__classifiers:
            context.chat_data.setdefault('CLASSIFIER', self.__default_classifier)
            update.message.reply_text(("Pass valid alias of classifier as argument to command.\n"
                                       f"Aliases: {', '.join(self.__classifiers.keys())}\n"
                                       "/get_classifiers to see description on each of them\n"
                                       f"Current classifier: {context.chat_data['CLASSIFIER']}"))
        else:
            context.chat_data['CLASSIFIER'] = new_value
            update.message.reply_text(f"Classifier is now set to: {context.chat_data['CLASSIFIER']}")
            logger.info("u: %d-%s c: %s | CLASSIFIER is set to %s",
                        update.effective_user.id, update.effective_user.username, update.effective_chat.id,
                        new_value)

    def get_classifiers(self, update: Update, context: CallbackContext) -> None:
        classifiers = '\n'.join([f" {k}: {v.description}" for k, v in self.__classifiers.items()])
        update.message.reply_text("Description for each alias of classifiers:\n" + classifiers +
                                  "\n\nTo set /set_classifier *classifier_alias*\n\n" +
                                  f"Current classifier: {context.chat_data['CLASSIFIER']}")


# there's a lot to improve here
def main():
    config = configparser.ConfigParser()
    config.read('config.ini')
    token = config.get('telegram', 'token', fallback=None)
    device = config.get('environment', 'device', fallback='cuda')

    # amount of question blocks, standard and generated questions in each block
    blocks = config.getint('bot', 'blocks', fallback=None)
    st_q = config.getint('bot', 'standard_questions', fallback=None)
    gen_q = config.getint('bot', 'generated_questions', fallback=None)

    # initialize classifiers and bind output function
    single_cls = SingleEmotionClassifier(device=device)
    multiple_cls = MultipleEmotionsClassifier(device=device)
    single_cls.output = single_cl_output
    multiple_cls.output = multiple_cl_output

    classifiers = {'single': single_cls, 'multiple': multiple_cls}

    # there is no checks in handlers for this list to be not empty for now
    # so it's better to ensure it has enough questions
    standard_questions = []
    dir_path = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(dir_path, 'standard_questions.txt'), 'r') as f:
        standard_questions = f.read().split('\n')

    persistence = PicklePersistence(filename='sessions_data', on_flush=False)
    bot = Bot(token, classifiers, standard_questions, 'multiple',
              blocks=blocks,
              st_q=st_q,
              gen_q=gen_q,
              persistence=persistence,
              timeout=datetime.timedelta(minutes=30))
    bot.run()


if __name__ == '__main__':
    main()
