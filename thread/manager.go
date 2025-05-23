package thread

import (
	"context"
	"github.com/jcooky/go-din"
	"log/slog"
	"strings"

	"github.com/habiliai/agentruntime/entity"
	myerrors "github.com/habiliai/agentruntime/errors"
	"github.com/habiliai/agentruntime/internal/db"
	"github.com/habiliai/agentruntime/internal/mylog"
	"github.com/habiliai/agentruntime/errors"
	"gorm.io/datatypes"
	"gorm.io/gorm"
)

type (
	Manager interface {
		CreateThread(ctx context.Context, instruction string) (*entity.Thread, error)
		AddMessage(ctx context.Context, threadId uint, sender string, content entity.MessageContent) (*entity.Message, error)
		GetMessages(ctx context.Context, threadId uint, order string, cursor uint, limit uint) ([]entity.Message, error)
		GetNumMessages(ctx context.Context, threadId uint) (int64, error)
		GetThreads(ctx context.Context, cursor uint, limit uint) ([]entity.Thread, error)
		GetThreadById(ctx context.Context, threadId uint) (*entity.Thread, error)
	}

	manager struct {
		logger *mylog.Logger
		db     *gorm.DB
	}
)

func (s *manager) GetNumMessages(ctx context.Context, threadId uint) (int64, error) {
	_, tx := db.OpenSession(ctx, s.db)

	var count int64
	if err := tx.Model(&entity.Message{}).Where("thread_id = ?", threadId).Count(&count).Error; err != nil {
		return 0, errors.Wrapf(err, "failed to count messages")
	}

	return count, nil
}

func (s *manager) GetThreadById(ctx context.Context, threadId uint) (*entity.Thread, error) {
	_, tx := db.OpenSession(ctx, s.db)

	var thread entity.Thread
	if err := tx.First(&thread, threadId).Error; err != nil {
		return nil, errors.Wrapf(err, "failed to find thread")
	}

	return &thread, nil
}

func (s *manager) GetThreads(ctx context.Context, cursor uint, limit uint) ([]entity.Thread, error) {
	_, tx := db.OpenSession(ctx, s.db)

	var threads []entity.Thread
	if err := tx.Where("id > ?", cursor).Order("id ASC").Limit(int(limit)).Find(&threads).Error; err != nil {
		return nil, errors.Wrapf(err, "failed to find threads")
	}

	return threads, nil
}

func (s *manager) GetMessages(
	ctx context.Context,
	threadId uint,
	order string,
	cursor uint,
	limit uint,
) (messages []entity.Message, err error) {
	_, tx := db.OpenSession(ctx, s.db)
	order = strings.ToUpper(order)
	if order != "ASC" && order != "DESC" {
		return nil, errors.Wrapf(myerrors.ErrInvalidParams, "invalid order")
	}

	stmt := tx.Model(&entity.Message{}).
		Where("thread_id = ?", threadId).
		Order("created_at " + order)

	if cursor != 0 {
		if order == "ASC" {
			stmt = stmt.Where("id > ?", cursor)
		} else {
			stmt = stmt.Where("id < ?", cursor)
		}
	}
	if limit == 0 {
		limit = 50
	}

	if err := stmt.Limit(int(limit)).Find(&messages).Error; err != nil {
		return nil, errors.Wrapf(err, "failed to find messages")
	}

	return
}

func (s *manager) AddMessage(ctx context.Context, threadId uint, sender string, content entity.MessageContent) (*entity.Message, error) {
	_, tx := db.OpenSession(ctx, s.db)
	var thread entity.Thread
	if r := tx.Find(&thread, threadId); r.Error != nil {
		return nil, errors.Wrapf(r.Error, "failed to find thread")
	} else if r.RowsAffected == 0 {
		return nil, errors.Wrapf(myerrors.ErrNotFound, "thread not found")
	}

	msg := entity.Message{
		ThreadID: thread.ID,
		Content:  datatypes.NewJSONType(content),
		User:     sender,
	}

	if err := tx.Save(&msg).Error; err != nil {
		return nil, errors.Wrapf(err, "failed to save message")
	}

	return &msg, nil
}

func (s *manager) CreateThread(ctx context.Context, instruction string) (*entity.Thread, error) {
	_, tx := db.OpenSession(ctx, s.db)

	thread := entity.Thread{
		Instruction: instruction,
	}

	if err := tx.Create(&thread).Error; err != nil {
		return nil, errors.Wrapf(err, "failed to create thread")
	}

	return &thread, nil
}

func init() {
	din.RegisterT(func(c *din.Container) (Manager, error) {
		logger, err := din.Get[*slog.Logger](c, mylog.Key)
		if err != nil {
			return nil, err
		}

		return &manager{
			logger: logger,
			db:     din.MustGet[*gorm.DB](c, db.Key),
		}, nil
	})
}
